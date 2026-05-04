# Session Log: Sweep Script Alignment & Cost Ratio Sweep

**Date:** 2026-01-28

## Overview

This session had two parts:

1. **Aligned the two sweep scripts** (`scripts/cost_matrix_sweep.py` and `playground/cost_sensitive_loss_classification/cost_ratio_sweep.py`) so they produce identical metric columns, graph outputs, and similar CLI interfaces. This required upstream changes to `utils/utils.py` and `train_spiders.py`.
2. **Ran the full 19-value cost ratio sweep** (cost values 1-10 by 1, then 20-100 by 10) on the binary spider classifier.

---

## 1. Code Changes

Four files were modified to achieve alignment between the parent repo sweep (`scripts/cost_matrix_sweep.py`, which uses HuggingFace Trainer) and the playground sweep (`cost_ratio_sweep.py`, which uses a custom training loop).

### `utils/utils.py`

Added new metrics to `perform_comprehensive_evaluation()`:

- **New imports:** `scipy.special.softmax`, `sklearn.metrics.balanced_accuracy_score`, `cohen_kappa_score`, `roc_auc_score`
- **New per-class metric:** `"support"` (sample count per class)
- **New overall metrics:** `eval_balanced_accuracy`, `eval_kappa`, `eval_auc`, `eval_recall_class0`, `eval_expected_cost`, `eval_prevalence_class0`
- **Expected cost computation:** Uses the config's `cost_matrix` and confusion matrix to compute weighted misclassification cost
- **Updated `save_metrics_to_file()`** to include all new metrics in both JSON and text outputs

### `scripts/cost_matrix_sweep.py`

- **New CLI arg:** `--seed` (default 42) — replaces hardcoded `set_seed(42)` in `make_objective()`
- **New metric extraction:** balanced_accuracy, kappa, auc, recall_class0, expected_cost, prevalence_class0, per-class support — stored as Optuna trial user attributes
- **New graphs (4):** `per_class_f1`, `expected_cost`, `precision_recall_tradeoff`, updated `per_class_fnr`
- **Updated `overall_metrics`:** Now 5 traces (AUC, Balanced Accuracy, Kappa, Accuracy, F1)
- **Updated dashboard:** From 2x2 to 2x3 layout (Overall Metrics, Per-Class Recall, Per-Class F1, Expected Cost, CM Cells, P-R Tradeoff)
- **Updated best trial printout** to include all new metrics

### `playground/.../train_spiders.py`

- **New imports:** `sklearn.metrics.accuracy_score`, `sklearn.metrics.f1_score`
- **Updated `eval_predictions_binary()`:** Added `accuracy`, `f1_score` (macro), per-class `false_positive_rate` and `false_negative_rate` to the return dict

### `playground/.../cost_ratio_sweep.py`

Full rewrite for alignment:

- **Renamed** Optuna param from `"cost_ratio"` to `"cost_value"` throughout
- **New CLI args:** `--min`, `--max`, `--step` for grid generation (alternative to `--values`)
- **Per-class metrics** renamed from `{metric}_{i}` to `class_{i}_{metric}` convention
- **New metrics extracted:** accuracy, f1_score, prevalence_class0, FPR, FNR, support
- **Added optuna built-in plots** (slice, history)
- **Added graphs:** per_class_fnr, per_class_f1, expected_cost, precision_recall_tradeoff
- **Grid generation** uses `np.arange` when `--values` not provided

---

## 2. Unified Column Schema

Both sweep scripts now produce CSV files with identical columns:

| Column | Description |
|---|---|
| `trial_number` | Optuna trial index |
| `cost_value` | The swept cost parameter |
| `objective_class_0_recall` | Primary objective (maximize) |
| `accuracy` | Overall accuracy |
| `balanced_accuracy` | Balanced accuracy (mean per-class recall) |
| `f1_score` | Macro-averaged F1 |
| `kappa` | Cohen's quadratic weighted kappa |
| `auc` | ROC AUC (binary: positive-class prob; multi: OVO weighted) |
| `recall_class0` | Same as objective (for convenience) |
| `expected_cost` | Cost-matrix-weighted misclassification cost |
| `loss` | Final validation loss |
| `prevalence_class0` | Class 0 proportion in evaluation set |
| `class_{i}_precision` | Per-class precision |
| `class_{i}_recall` | Per-class recall |
| `class_{i}_f1_score` | Per-class F1 |
| `class_{i}_false_positive_rate` | Per-class FPR |
| `class_{i}_false_negative_rate` | Per-class FNR |
| `class_{i}_support` | Per-class sample count |
| `cm_{i}_{j}` | Confusion matrix cell (true=i, pred=j) |

---

## 3. Unified Graph Set

Both scripts generate 10 graph pairs (HTML interactive + PNG static) in `graphs/`:

| Graph | Description |
|---|---|
| `optuna_slice` | Optuna built-in: cost_value vs objective |
| `optuna_history` | Optuna built-in: optimization history |
| `overall_metrics` | AUC, Balanced Accuracy, Kappa, Accuracy, F1 vs cost_value |
| `per_class_recall` | Per-class recall vs cost_value |
| `per_class_f1` | Per-class F1 vs cost_value |
| `per_class_fnr` | Per-class FNR vs cost_value |
| `expected_cost` | Expected misclassification cost vs cost_value |
| `confusion_matrix_cells` | CM cell counts vs cost_value |
| `precision_recall_tradeoff` | Class 0 precision vs recall across cost values |
| `dashboard` | 2x3 combined subplot (all key charts) |

---

## 4. Sweep Configuration

```json
{
  "grid_values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
  "n_trials": 19,
  "base_loss": "ce",
  "lambd": 10.0,
  "n_epochs": 30,
  "patience": 5,
  "seed": 42,
  "output_dir": "results/sweep_cost_ratio"
}
```

Command used:
```bash
cd <repo-root>/playground/cost_sensitive_loss_classification
micromamba run -n ml python cost_ratio_sweep.py \
    --values "1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100"
```

---

## 5. Sweep Results

19 trials, ResNet-50 on binary spider dataset (Black Widow=class 0 vs False Widow=class 1). Sorted by cost_value.

| Cost Value | Accuracy | Balanced Acc | AUC | Kappa | Class 0 Recall | Class 1 Recall | Expected Cost | Loss |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 88.86% | 88.85% | 0.9623 | 0.7772 | 96.89% | 80.80% | 0.0156 | 0.645 |
| 2.0 | 85.52% | 85.50% | 0.9562 | 0.7103 | 96.00% | 75.00% | 0.0401 | 1.032 |
| 3.0 | 85.30% | 85.28% | 0.9576 | 0.7059 | 94.22% | 76.34% | 0.0869 | 1.408 |
| 4.0 | 81.74% | 81.70% | 0.9568 | 0.6345 | 96.44% | 66.96% | 0.0713 | 1.473 |
| 5.0 | 67.93% | 67.87% | 0.8695 | 0.3578 | 95.56% | 40.18% | 0.1114 | 2.564 |
| 6.0 | 82.18% | 82.15% | 0.9473 | 0.6434 | 97.78% | 66.52% | 0.0668 | 1.694 |
| 7.0 | 79.73% | 79.69% | 0.9552 | 0.5943 | 98.22% | 61.16% | 0.0624 | 1.613 |
| 8.0 | 84.63% | 84.61% | 0.9561 | 0.6925 | 96.00% | 73.21% | 0.1604 | 2.420 |
| 9.0 | 76.17% | 76.12% | 0.9459 | 0.5229 | 98.67% | 53.57% | 0.0601 | 2.152 |
| 10.0 | 80.40% | 80.37% | 0.9502 | 0.6077 | 96.44% | 64.29% | 0.1782 | 2.757 |
| 20.0 | 76.39% | 76.35% | 0.9453 | 0.5274 | 96.44% | 56.25% | 0.3563 | 4.265 |
| 30.0 | 73.27% | 73.22% | 0.9480 | 0.4649 | 99.11% | 47.32% | 0.1336 | 3.447 |
| 40.0 | 66.59% | 66.52% | 0.9415 | 0.3309 | 99.11% | 33.93% | 0.1782 | 4.418 |
| 50.0 | 60.58% | 60.49% | 0.9232 | 0.2102 | **100.00%** | 20.98% | 0.0000 | 2.706 |
| 60.0 | 67.48% | 67.41% | 0.9575 | 0.3487 | 99.56% | 35.27% | 0.1336 | 3.626 |
| 70.0 | 68.37% | 68.31% | 0.9396 | 0.3666 | 99.11% | 37.50% | 0.3118 | 5.495 |
| **80.0** | **50.11%** | **50.00%** | 0.8485 | 0.0000 | **100.00%** | **0.00%** | 0.0000 | 3.367 |
| 90.0 | 54.34% | 54.24% | 0.8976 | 0.0850 | 99.56% | 8.93% | 0.2004 | 6.714 |
| **100.0** | **50.11%** | **50.00%** | 0.8664 | 0.0000 | **100.00%** | **0.00%** | 0.0000 | 3.389 |

---

## 6. Key Findings

### 1. Best overall quality: cost_value=1.0

| Metric | Value |
|---|---|
| Accuracy | 88.86% |
| Balanced Accuracy | 88.85% |
| AUC | 0.9623 |
| Kappa | 0.7772 |
| Class 0 Recall | 96.89% |
| Class 1 Recall | 80.80% |
| Expected Cost | 0.0156 |

Cost=1.0 achieves the best balance of all metrics. Class 0 (Black Widow) recall is already 96.89% while maintaining strong class 1 performance.

### 2. Safety-focused sweet spot: cost_value=9.0

For applications prioritizing black widow detection over overall accuracy:
- Class 0 Recall: 98.67% (only 3 black widows missed out of 225)
- AUC: 0.9459 (still strong discriminative ability)
- Accuracy: 76.17% (significant tradeoff)

### 3. Diminishing returns in the 2-10 range

Increasing cost from 1 to 10 does not monotonically improve class 0 recall. The relationship is noisy:
- Cost 1: 96.89% recall
- Cost 5: 95.56% recall (worse than cost 1)
- Cost 7: 98.22% recall
- Cost 9: 98.67% recall (best in this range)

### 4. Degenerate behavior at cost >= 80

At cost values 80 and 100, the model collapses to predicting all samples as class 0:
- Class 0 Recall: 100% (trivially)
- Class 1 Recall: 0%
- Kappa: 0.0
- Accuracy: 50.11% (equal to class 0 prevalence)

The model has learned that the cost penalty for missing a black widow is so extreme that it is safest to never predict class 1. This is the expected degenerate case for excessive cost-sensitive penalization.

### 5. Expected cost is not a reliable standalone metric

Expected cost reaches 0.0 at cost=50, 80, and 100, but this is misleading -- it is zero because the model never predicts class 1, so it never incurs the `M[0][1]` penalty. The expected cost metric is only meaningful when the model makes non-trivial predictions for both classes.

---

## 7. Artifacts Produced

### Code changes

| File | Change Type |
|---|---|
| `utils/utils.py` | Modified (new metrics in evaluation pipeline) |
| `scripts/cost_matrix_sweep.py` | Modified (new graphs, metrics, dashboard, --seed) |
| `playground/.../train_spiders.py` | Modified (accuracy, f1, FPR, FNR) |
| `playground/.../cost_ratio_sweep.py` | Rewritten (full alignment with parent repo) |

### Sweep outputs

```
playground/cost_sensitive_loss_classification/results/sweep_cost_ratio/
├── sweep_config.json
├── sweep_results.csv
├── cost_ratio_sweep.duckdb
└── graphs/
    ├── optuna_slice.html/.png
    ├── optuna_history.html/.png
    ├── overall_metrics.html/.png
    ├── per_class_recall.html/.png
    ├── per_class_f1.html/.png
    ├── per_class_fnr.html/.png
    ├── expected_cost.html/.png
    ├── confusion_matrix_cells.html/.png
    ├── precision_recall_tradeoff.html/.png
    └── dashboard.html/.png
```

### Experiment checkpoints

19 checkpoints saved to `playground/cost_sensitive_loss_classification/experiments/sweep_cr_*`.

---

## 8. How to Reproduce

```bash
cd <repo-root>/playground/cost_sensitive_loss_classification
micromamba run -n ml python cost_ratio_sweep.py \
    --values "1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100"
```

To run with different grid parameters:
```bash
# Fine-grained sweep from 0.5 to 5.0 in steps of 0.5
micromamba run -n ml python cost_ratio_sweep.py --min 0.5 --max 5.0 --step 0.5

# Custom values
micromamba run -n ml python cost_ratio_sweep.py --values "0.5,1,2,5,10,50"
```

### Querying results with DuckDB

```python
import duckdb
con = duckdb.connect("results/sweep_cost_ratio/cost_ratio_sweep.duckdb")
con.sql("""
    SELECT cost_value, accuracy, balanced_accuracy, auc, kappa,
           class_0_recall, class_1_recall, expected_cost
    FROM sweep_results
    ORDER BY cost_value
""").show()
```
