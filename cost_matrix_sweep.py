"""
Cost Matrix Sweep — Optuna GridSampler + DuckDB + Polars + Plotly

Sweeps a single cost matrix cell across a grid of values, trains the model
for each, and collects comprehensive metrics.  Results are persisted to a
DuckDB database and visualised with interactive Plotly charts.

Usage:
    micromamba activate ml
    python cost_matrix_sweep.py --config config/2classSpiders.json

The script reads sweep parameters from command-line flags (or uses sensible
defaults targeting cell [0][1] with range 0–10, step 0.5).
"""

import argparse
import copy
import json
import os
import sys

import duckdb
import numpy as np
import polars as pl
import optuna
from optuna.samplers import GridSampler
from pathlib import Path
from transformers import set_seed

# ---------------------------------------------------------------------------
# Make sure repo root is on sys.path (mirrors train.py lines 10-13)
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, parent_dir)

from train import main as train_main
from utils.utils import ScriptTrainingArguments

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_sweep_args():
    parser = argparse.ArgumentParser(
        description="Cost matrix cell sweep with Optuna GridSampler"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the base training config JSON (e.g. config/2classSpiders.json)"
    )
    parser.add_argument("--row", type=int, default=0, help="Cost matrix row to sweep")
    parser.add_argument("--col", type=int, default=1, help="Cost matrix column to sweep")
    parser.add_argument("--min", type=float, default=0.0, dest="cost_min", help="Minimum cost value")
    parser.add_argument("--max", type=float, default=10.0, dest="cost_max", help="Maximum cost value")
    parser.add_argument("--step", type=float, default=0.5, help="Step size between cost values")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/sweep_cost_{row}_{col})"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Build ScriptTrainingArguments from a JSON config dict
# ---------------------------------------------------------------------------

def build_script_args(config_dict: dict) -> ScriptTrainingArguments:
    """Build a ScriptTrainingArguments from a plain dict without argparse."""
    from dataclasses import fields as dataclass_fields

    field_names = {f.name for f in dataclass_fields(ScriptTrainingArguments)}
    filtered = {}
    for k, v in config_dict.items():
        if k in field_names:
            # cost_matrix needs to stay as a Python list (not JSON string)
            if k == "cost_matrix" and isinstance(v, list):
                filtered[k] = json.dumps(v)
            else:
                filtered[k] = v

    from transformers import HfArgumentParser
    hf_parser = HfArgumentParser(ScriptTrainingArguments)
    parsed = hf_parser.parse_dict(filtered)[0]

    if parsed.cost_matrix is not None:
        parsed.cost_matrix = json.loads(parsed.cost_matrix)

    return parsed


# ---------------------------------------------------------------------------
# Read metrics JSON produced by perform_comprehensive_evaluation
# ---------------------------------------------------------------------------

def read_trial_metrics(results_dir: str) -> dict:
    """Read the metrics JSON from a training run's results directory."""
    results_path = Path(results_dir)
    # Find the metrics JSON file (pattern: metrics_*_test.json)
    json_files = list(results_path.glob("metrics_*_test.json"))
    if not json_files:
        # Fall back to any JSON with "metrics" in the name
        json_files = list(results_path.glob("metrics_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No metrics JSON found in {results_dir}")

    with open(json_files[0]) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(base_config: dict, row: int, col: int):
    """Return an Optuna objective function closed over the base config."""

    def objective(trial: optuna.Trial) -> float:
        cost_value = trial.suggest_float("cost_value", 0.0, 100.0)

        # Deep-copy config and set the cost matrix cell
        config = copy.deepcopy(base_config)
        config["cost_matrix"][row][col] = cost_value

        # Build ScriptTrainingArguments
        script_args = build_script_args(config)

        # Run full training + evaluation pipeline
        set_seed(42)
        results_dir = train_main(script_args)

        # Read metrics
        metrics = read_trial_metrics(results_dir)

        overall = metrics.get("overall_metrics", {})
        per_class = metrics.get("per_class_metrics", {})
        confusion = metrics.get("confusion_matrix", [])

        # Store all metrics as user attributes for later export
        trial.set_user_attr("results_dir", str(results_dir))
        trial.set_user_attr("accuracy", overall.get("accuracy", 0.0))
        trial.set_user_attr("f1_score", overall.get("f1_score", 0.0))
        trial.set_user_attr("loss", overall.get("loss", 0.0))

        # Per-class metrics (keyed by class index as string in the JSON)
        num_classes = len(confusion) if confusion else 0
        for cls_idx in range(num_classes):
            cls_key = str(cls_idx)
            for metric_name in ("accuracy", "precision", "recall", "f1_score",
                                "false_positive_rate", "false_negative_rate"):
                val = per_class.get(metric_name, {}).get(cls_key, 0.0)
                trial.set_user_attr(f"class_{cls_idx}_{metric_name}", val)

        # Confusion matrix cells
        for i, row_vals in enumerate(confusion):
            for j, cell_val in enumerate(row_vals):
                trial.set_user_attr(f"cm_{i}_{j}", cell_val)

        # Primary objective: class 0 recall (we want to maximise detection of
        # black widows — minimise FNR for class 0)
        class_0_recall = per_class.get("recall", {}).get("0", 0.0)

        print(f"\n[Trial {trial.number}] cost_value={cost_value:.2f}  "
              f"accuracy={overall.get('accuracy', 0):.4f}  "
              f"class_0_recall={class_0_recall:.4f}\n")

        return class_0_recall

    return objective


# ---------------------------------------------------------------------------
# Export results to DuckDB + Polars
# ---------------------------------------------------------------------------

def export_results(study: optuna.Study, output_dir: str, row: int, col: int):
    """Export all trial results to a Polars DataFrame and DuckDB database."""
    records = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        record = {
            "trial_number": trial.number,
            "cost_value": trial.params["cost_value"],
            "objective_class_0_recall": trial.value,
        }
        record.update(trial.user_attrs)
        # Remove results_dir from the record (not useful in the DB)
        record.pop("results_dir", None)
        records.append(record)

    if not records:
        print("WARNING: No completed trials to export.")
        return None

    df = pl.DataFrame(records)

    # Sort by cost_value for cleaner output
    df = df.sort("cost_value")

    # Save CSV
    csv_path = os.path.join(output_dir, "sweep_results.csv")
    df.write_csv(csv_path)
    print(f"Results CSV saved to: {csv_path}")

    # Save to DuckDB
    db_path = os.path.join(output_dir, "cost_matrix_sweep.duckdb")
    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS sweep_results")
    con.execute("CREATE TABLE sweep_results AS SELECT * FROM df")
    con.close()
    print(f"DuckDB database saved to: {db_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 80)
    summary_cols = ["cost_value", "accuracy", "f1_score",
                    "class_0_recall", "class_1_recall",
                    "class_0_false_negative_rate", "class_1_false_negative_rate"]
    available = [c for c in summary_cols if c in df.columns]
    print(df.select(available))
    print("=" * 80)

    return df


# ---------------------------------------------------------------------------
# Plotly visualisations
# ---------------------------------------------------------------------------

def generate_visualizations(study: optuna.Study, df: pl.DataFrame,
                            output_dir: str, row: int, col: int):
    """Generate interactive Plotly charts and Optuna built-in visualisations."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    graphs_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    # ---- 1. Optuna built-in: slice plot (cost_value vs objective) ----
    try:
        from optuna.visualization import plot_slice, plot_optimization_history
        fig_slice = plot_slice(study, params=["cost_value"])
        fig_slice.write_html(os.path.join(graphs_dir, "optuna_slice.html"))
        fig_slice.write_image(os.path.join(graphs_dir, "optuna_slice.png"))

        fig_hist = plot_optimization_history(study)
        fig_hist.write_html(os.path.join(graphs_dir, "optuna_history.html"))
        fig_hist.write_image(os.path.join(graphs_dir, "optuna_history.png"))
        print("Optuna built-in visualizations saved.")
    except Exception as e:
        print(f"WARNING: Could not generate Optuna built-in plots: {e}")

    # ---- 2. Overall accuracy & F1 vs cost value ----
    cost_vals = df["cost_value"].to_list()

    fig_overall = go.Figure()
    if "accuracy" in df.columns:
        fig_overall.add_trace(go.Scatter(
            x=cost_vals, y=df["accuracy"].to_list(),
            mode="lines+markers", name="Accuracy"))
    if "f1_score" in df.columns:
        fig_overall.add_trace(go.Scatter(
            x=cost_vals, y=df["f1_score"].to_list(),
            mode="lines+markers", name="F1 Score"))
    fig_overall.update_layout(
        title=f"Overall Metrics vs Cost Matrix [{row}][{col}]",
        xaxis_title="Cost Value", yaxis_title="Score",
        template="plotly_white")
    fig_overall.write_html(os.path.join(graphs_dir, "overall_metrics.html"))
    fig_overall.write_image(os.path.join(graphs_dir, "overall_metrics.png"))

    # ---- 3. Per-class recall vs cost value ----
    fig_recall = go.Figure()
    for cls_idx in range(2):
        col_name = f"class_{cls_idx}_recall"
        if col_name in df.columns:
            fig_recall.add_trace(go.Scatter(
                x=cost_vals, y=df[col_name].to_list(),
                mode="lines+markers", name=f"Class {cls_idx} Recall"))
    fig_recall.update_layout(
        title=f"Per-Class Recall vs Cost Matrix [{row}][{col}]",
        xaxis_title="Cost Value", yaxis_title="Recall",
        template="plotly_white")
    fig_recall.write_html(os.path.join(graphs_dir, "per_class_recall.html"))
    fig_recall.write_image(os.path.join(graphs_dir, "per_class_recall.png"))

    # ---- 4. Per-class FNR vs cost value ----
    fig_fnr = go.Figure()
    for cls_idx in range(2):
        col_name = f"class_{cls_idx}_false_negative_rate"
        if col_name in df.columns:
            fig_fnr.add_trace(go.Scatter(
                x=cost_vals, y=df[col_name].to_list(),
                mode="lines+markers", name=f"Class {cls_idx} FNR"))
    fig_fnr.update_layout(
        title=f"Per-Class False Negative Rate vs Cost Matrix [{row}][{col}]",
        xaxis_title="Cost Value", yaxis_title="FNR",
        template="plotly_white")
    fig_fnr.write_html(os.path.join(graphs_dir, "per_class_fnr.html"))
    fig_fnr.write_image(os.path.join(graphs_dir, "per_class_fnr.png"))

    # ---- 5. Confusion matrix cell counts vs cost value ----
    fig_cm = go.Figure()
    for i in range(2):
        for j in range(2):
            cm_col = f"cm_{i}_{j}"
            if cm_col in df.columns:
                label = f"True {i} → Pred {j}"
                fig_cm.add_trace(go.Scatter(
                    x=cost_vals, y=df[cm_col].to_list(),
                    mode="lines+markers", name=label))
    fig_cm.update_layout(
        title=f"Confusion Matrix Cells vs Cost Matrix [{row}][{col}]",
        xaxis_title="Cost Value", yaxis_title="Count",
        template="plotly_white")
    fig_cm.write_html(os.path.join(graphs_dir, "confusion_matrix_cells.html"))
    fig_cm.write_image(os.path.join(graphs_dir, "confusion_matrix_cells.png"))

    # ---- 6. Comprehensive dashboard (subplots) ----
    fig_dash = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Overall Accuracy & F1",
            "Per-Class Recall",
            "Per-Class FNR",
            "Confusion Matrix Cells",
        ))

    # Overall
    if "accuracy" in df.columns:
        fig_dash.add_trace(go.Scatter(
            x=cost_vals, y=df["accuracy"].to_list(),
            mode="lines+markers", name="Accuracy"), row=1, col=1)
    if "f1_score" in df.columns:
        fig_dash.add_trace(go.Scatter(
            x=cost_vals, y=df["f1_score"].to_list(),
            mode="lines+markers", name="F1"), row=1, col=1)

    # Recall
    for cls_idx in range(2):
        col_name = f"class_{cls_idx}_recall"
        if col_name in df.columns:
            fig_dash.add_trace(go.Scatter(
                x=cost_vals, y=df[col_name].to_list(),
                mode="lines+markers", name=f"Cls {cls_idx} Recall"), row=1, col=2)

    # FNR
    for cls_idx in range(2):
        col_name = f"class_{cls_idx}_false_negative_rate"
        if col_name in df.columns:
            fig_dash.add_trace(go.Scatter(
                x=cost_vals, y=df[col_name].to_list(),
                mode="lines+markers", name=f"Cls {cls_idx} FNR"), row=2, col=1)

    # CM cells
    for i in range(2):
        for j in range(2):
            cm_col = f"cm_{i}_{j}"
            if cm_col in df.columns:
                fig_dash.add_trace(go.Scatter(
                    x=cost_vals, y=df[cm_col].to_list(),
                    mode="lines+markers", name=f"T{i}→P{j}"), row=2, col=2)

    fig_dash.update_layout(
        title_text=f"Cost Matrix [{row}][{col}] Sweep Dashboard",
        height=800, template="plotly_white", showlegend=True)
    fig_dash.write_html(os.path.join(graphs_dir, "dashboard.html"))
    fig_dash.write_image(os.path.join(graphs_dir, "dashboard.png"))

    print(f"All visualizations saved to: {graphs_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_sweep_args()

    # Load base config
    with open(args.config) as f:
        base_config = json.load(f)

    row, col = args.row, args.col

    # Validate cost matrix dimensions
    cm = base_config.get("cost_matrix")
    if cm is None:
        raise ValueError("Config must include a cost_matrix field")
    if row >= len(cm) or col >= len(cm[0]):
        raise ValueError(
            f"Cell [{row}][{col}] out of bounds for "
            f"{len(cm)}x{len(cm[0])} cost matrix"
        )

    # Build grid values
    grid_values = list(np.arange(args.cost_min, args.cost_max + args.step / 2, args.step))
    grid_values = [round(v, 4) for v in grid_values]
    n_trials = len(grid_values)

    # Output directory
    output_dir = args.output_dir or f"results/sweep_cost_{row}_{col}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("COST MATRIX SWEEP — Optuna GridSampler")
    print("=" * 60)
    print(f"Config:        {args.config}")
    print(f"Cell:          [{row}][{col}]")
    print(f"Range:         {args.cost_min} to {args.cost_max}, step {args.step}")
    print(f"Grid values:   {grid_values}")
    print(f"Total trials:  {n_trials}")
    print(f"Output:        {output_dir}")
    print("=" * 60)

    # Save sweep configuration
    sweep_meta = {
        "config": args.config,
        "row": row, "col": col,
        "cost_min": args.cost_min, "cost_max": args.cost_max,
        "step": args.step, "grid_values": grid_values,
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "sweep_config.json"), "w") as f:
        json.dump(sweep_meta, f, indent=2)

    # Create Optuna study with GridSampler
    sampler = GridSampler({"cost_value": grid_values})
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"cost_matrix_{row}_{col}_sweep",
    )

    objective = make_objective(base_config, row, col)
    study.optimize(objective, n_trials=n_trials)

    # Export results
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)
    df = export_results(study, output_dir, row, col)

    # Generate visualizations
    if df is not None:
        print("\nGenerating visualizations...")
        generate_visualizations(study, df, output_dir, row, col)

    # Print best trial
    best = study.best_trial
    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    print(f"Trial:           #{best.number}")
    print(f"Cost value:      {best.params['cost_value']:.4f}")
    print(f"Class 0 Recall:  {best.value:.4f}")
    print(f"Accuracy:        {best.user_attrs.get('accuracy', 'N/A')}")
    print(f"F1 Score:        {best.user_attrs.get('f1_score', 'N/A')}")
    print(f"Results dir:     {best.user_attrs.get('results_dir', 'N/A')}")
    print("=" * 60)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"  - sweep_results.csv")
    print(f"  - cost_matrix_sweep.duckdb")
    print(f"  - graphs/ (interactive HTML + static PNG)")


if __name__ == "__main__":
    main()
