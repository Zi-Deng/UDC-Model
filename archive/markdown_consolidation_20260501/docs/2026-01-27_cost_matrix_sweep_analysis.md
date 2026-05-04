# Session Log: Cost Matrix Sweep Analysis

**Date:** 2026-01-27

## Overview

This session researched optimal approaches for a 1D cost matrix ablation study, implemented an Optuna-based sweep script (`scripts/cost_matrix_sweep.py`) with DuckDB persistence and Polars dataframes, and ran a sweep of `cost_matrix[0][1]` from 0 to 10 (step 1) to analyze the effect of penalizing misclassification of black widows (class 0) as false widows (class 1).

---

## 1. New Dependency

```bash
pip install kaleido  # installed v1.2.0
```

Kaleido is required by Plotly for exporting static PNG images from interactive charts via `write_image()`.

---

## 2. Research: Optimal Approach for 1D Cost Matrix Ablation

Evaluated six tools/frameworks for this use case:

| Tool | Approach | Verdict |
|---|---|---|
| **Optuna (GridSampler)** | Grid search with uniform coverage | **Selected** — best fit for 1D ablation |
| Ray Tune | Distributed HPO, heavy infrastructure | Overkill for single-parameter sweep |
| W&B Sweeps | Cloud-hosted sweep management | Requires W&B account, unnecessary complexity |
| Ax (Meta) | Bayesian optimization via BoTorch | Concentrates samples near optimum, bad for landscape analysis |
| SALib | Sensitivity analysis (Sobol, Morris) | Designed for multi-parameter sensitivity, not single-cell sweep |
| ABLATOR | Ablation study framework | Early-stage, limited documentation |

**Key finding:** For single-parameter ablation studies, grid search is the correct strategy. Bayesian optimization (TPE, GP) concentrates samples near the optimum, which hurts the ability to understand the full landscape — the opposite of what an ablation study needs. Optuna's `GridSampler` provides uniform coverage while still leveraging Optuna's trial tracking, user attributes, and built-in visualization infrastructure.

---

## 3. Files Created

### `scripts/cost_matrix_sweep.py`

Standalone Optuna-based cost matrix sweep script. Key design decisions:

- **Does NOT use `Trainer.hyperparameter_search()`** because `cost_matrix` is not a `TrainingArguments` field — it's a custom parameter in `ScriptTrainingArguments`.
- **Uses Optuna in-memory storage** (Optuna doesn't natively support DuckDB), then exports to DuckDB after completion.
- **Uses `build_script_args()`** to construct `ScriptTrainingArguments` from a dict without going through argparse.
- **Calls `train.main(script_args)` directly** for each trial.
- **Primary objective:** Class 0 recall (maximize black widow detection).
- **Records all metrics** (accuracy, F1, per-class recall/precision/FPR/FNR, confusion matrix cells) as Optuna trial user attributes for post-hoc analysis.
- **Uses Polars** for all DataFrame operations and **DuckDB** for database persistence, per project conventions.
- **Generates Plotly visualizations:** Optuna built-in slice/history plots + custom charts (overall metrics, per-class recall, FNR, confusion matrix cells, dashboard).

**Usage:**
```bash
micromamba activate ml
python scripts/cost_matrix_sweep.py --config config/2classSpiders.json

# With custom parameters:
python scripts/cost_matrix_sweep.py --config config/2classSpiders.json \
    --row 0 --col 1 --min 0.0 --max 10.0 --step 1 \
    --output-dir results/sweep_cost_0_1
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--config` | *(required)* | Path to base training config JSON |
| `--row` | 0 | Cost matrix row to sweep |
| `--col` | 1 | Cost matrix column to sweep |
| `--min` | 0.0 | Minimum cost value |
| `--max` | 10.0 | Maximum cost value |
| `--step` | 0.5 | Step size between cost values |
| `--output-dir` | `results/sweep_cost_{row}_{col}` | Output directory |

### `config/sweep_2class_bw_cost.json`

Sweep configuration file for the existing bash-based sweep system (`run_cost_matrix_sweep.sh`), targeting `cost_matrix[0][1]`:

```json
{
    "cost_range": { "min": 0.0, "max": 10.0, "step": 0.5 },
    "matrix_cell": { "row": 0, "col": 1 },
    "experiment": {
        "output_dir": "results/sweep_bw_misclass_cost_0_1",
        "base_config": "config/2classSpiders.json"
    },
    "analysis": { "update_graphs_each_iteration": true, "generate_final_report": true }
}
```

---

## 4. Sweep Results

**Configuration:** `cost_matrix[0][1]` swept from 0 to 10, step 1 (11 trials). Base config: optimized HPO parameters from previous session (lr=0.0003, batch=32, frozen_stages=3, early_stopping_patience=5, epochs=30).

**Dataset:** 2-class spider classification — Latrodectus hesperus (class 0, black widow) vs Steatoda grossa (class 1, false widow). Train=2429, Val=270, Test=300.

### Full Results Table

| Cost Value | Accuracy | Class 0 Recall | Class 1 Recall | FNR (Class 0) | FNR (Class 1) | CM: TP₀ | CM: FN₀ | CM: FP₀ | CM: TN₀ |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 87.67% | 88.67% | 86.67% | 11.33% | 13.33% | 133 | 17 | 20 | 130 |
| **1.0** | **89.33%** | **94.67%** | **84.00%** | **5.33%** | **16.00%** | **142** | **8** | **24** | **126** |
| 2.0 | 88.67% | 92.00% | 85.33% | 8.00% | 14.67% | 138 | 12 | 22 | 128 |
| 3.0 | 87.67% | 91.33% | 84.00% | 8.67% | 16.00% | 137 | 13 | 24 | 126 |
| 4.0 | 88.67% | 92.00% | 85.33% | 8.00% | 14.67% | 138 | 12 | 22 | 128 |
| 5.0 | 88.33% | 90.67% | 86.00% | 9.33% | 14.00% | 136 | 14 | 21 | 129 |
| 6.0 | 89.00% | 90.67% | 87.33% | 9.33% | 12.67% | 136 | 14 | 19 | 131 |
| 7.0 | 85.33% | 93.33% | 77.33% | 6.67% | 22.67% | 140 | 10 | 34 | 116 |
| 8.0 | 85.67% | 93.33% | 78.00% | 6.67% | 22.00% | 140 | 10 | 33 | 117 |
| 9.0 | 86.00% | 90.67% | 81.33% | 9.33% | 18.67% | 136 | 14 | 28 | 122 |
| 10.0 | 85.33% | 89.33% | 81.33% | 10.67% | 18.67% | 134 | 16 | 28 | 122 |

---

## 5. Key Findings

### 1. Optimal cost value: **1.0**

Cost=1.0 achieved the best combination of class 0 recall and overall accuracy:
- **Class 0 recall: 94.67%** (up from 88.67% baseline at cost=0) — a **+6.0 percentage point improvement**
- **Overall accuracy: 89.33%** (up from 87.67% baseline)
- **Class 0 FNR: 5.33%** (down from 11.33% baseline — more than halved)
- Only 8 black widows misclassified as false widows (down from 17)

### 2. Non-monotonic relationship

The relationship between cost value and class 0 recall is **not monotonic**. Increasing the cost beyond 1.0 does not continuously improve black widow detection:

- Cost 1.0: 94.67% recall (best)
- Cost 2.0-6.0: 90-92% recall (moderate improvement over baseline)
- Cost 7.0-8.0: 93.33% recall (second-best, but overall accuracy drops to ~85%)
- Cost 9.0-10.0: ~90% recall with degraded overall accuracy (85-86%)

### 3. High costs degrade overall performance

At cost values 7.0+, the model over-corrects — it successfully catches more black widows but at the expense of dramatically increasing false positives for class 0 (misclassifying false widows as black widows):

- Cost=0: 20 false positives for class 0
- Cost=7: 34 false positives for class 0 (+70%)
- Cost=10: 28 false positives for class 0 (+40%)

### 4. The cost-accuracy tradeoff

There is a clear tradeoff between class 0 recall and overall accuracy. Cost=1.0 is the **Pareto-optimal** point — it improves both metrics simultaneously. All other cost values either improve class 0 recall at the expense of accuracy, or don't improve recall enough to justify the accuracy loss.

---

## 6. Outputs Generated

All results saved to `results/sweep_cost_0_1/`:

```
results/sweep_cost_0_1/
├── sweep_results.csv              # Polars DataFrame export (all 11 trials)
├── cost_matrix_sweep.duckdb       # DuckDB database for post-hoc queries
├── sweep_config.json              # Sweep configuration metadata
└── graphs/
    ├── optuna_slice.html/.png     # Optuna built-in: cost_value vs objective
    ├── optuna_history.html/.png   # Optuna built-in: optimization history
    ├── overall_metrics.html/.png  # Accuracy & F1 vs cost value
    ├── per_class_recall.html/.png # Per-class recall vs cost value
    ├── per_class_fnr.html/.png    # Per-class FNR vs cost value
    ├── confusion_matrix_cells.html/.png  # CM cell counts vs cost value
    └── dashboard.html/.png        # Combined 2x2 subplot dashboard
```

### Querying results with DuckDB

```python
import duckdb
con = duckdb.connect("results/sweep_cost_0_1/cost_matrix_sweep.duckdb")
con.sql("SELECT cost_value, accuracy, class_0_recall, class_1_recall FROM sweep_results ORDER BY cost_value").show()
```

---

## 7. Convention Notes

This session established the following conventions for new scripts in the project:

- **Polars** for all DataFrame operations (not pandas)
- **DuckDB** for all SQL database persistence (not SQLite)
- These conventions are documented in `CLAUDE.md` and apply to all future scripts
