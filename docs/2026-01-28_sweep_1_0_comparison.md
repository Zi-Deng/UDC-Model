# Cost Matrix [1][0] Sweep + 3-Way Comparison Analysis

**Date:** 2026-01-28

## Overview

Extended the previous [0][1] analysis to the [1][0] cost matrix cell, which penalizes misclassifying **False Widows (class 1) as Black Widows (class 0)** — the "safe" error direction. This analysis runs the same 19-value cost sweep across all three pipelines and compares stability and performance.

**Cost Values Tested:** 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100

---

## 1. Modifications Made

### 1.1 Playground Script Updates

Modified `playground/cost_sensitive_loss_classification/train_spiders.py`:
- Added `--cost_row` and `--cost_col` CLI arguments
- Updated `build_cost_matrix()` to accept `row` and `col` parameters
- Updated `get_spider_criterion()` to pass row/col through

Modified `playground/cost_sensitive_loss_classification/cost_ratio_sweep.py`:
- Added `--row` and `--col` CLI arguments
- Updated `build_train_args()` to include `cost_row` and `cost_col`
- Updated all visualization titles to show dynamic `M[row][col]`
- Updated sweep metadata to include row/col

---

## 2. Sweep Execution

### 2.1 Parent [1][0] — Already Existed
- Location: `results/sweep_cost_1_0/sweep_results.csv`
- Trials: 28 (included all 19 requested values)

### 2.2 Playground [1][0]
```bash
cd playground/cost_sensitive_loss_classification
micromamba run -n ml python cost_ratio_sweep.py \
    --row 1 --col 0 \
    --values "1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100" \
    --output-dir results/sweep_cost_1_0
```
- Trials: 19 completed

### 2.3 Hybrid [1][0]
```bash
cd <repo-root>
micromamba run -n ml python scripts/cost_matrix_sweep.py \
    --config config/2classSpiders_reg.json \
    --row 1 --col 0 \
    --values "1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100" \
    --output-dir results/sweep_cost_1_0_reg
```
- Trials: 19 completed

---

## 3. Key Results

### 3.1 Collapse Behavior — Inverse of [0][1]

| Sweep | [0][1] Behavior | [1][0] Behavior |
|-------|-----------------|-----------------|
| Playground | Collapses to all-Black-Widow (C0 recall → 100%) | Collapses to all-False-Widow (C0 recall → 0%) |
| Collapsed costs | 80, 90, 100 | 7, 50, 80, 90, 100 |
| Min accuracy | 50.1% | 49.9% |
| Parent | No collapse | No collapse |
| Hybrid | No collapse | No collapse |

### 3.2 Stability Analysis

**Spread (max - min across all cost values):**

| Metric | Parent | Playground | Hybrid | Most Stable |
|--------|--------|------------|--------|-------------|
| accuracy | 0.0433 | 0.3697 | 0.0600 | Parent |
| class_0_recall | 0.0800 | 0.7822 | 0.1267 | Parent |
| class_1_recall | 0.0667 | 0.0625 | 0.0600 | **Hybrid** |
| f1_score | 0.0433 | 0.5348 | 0.0607 | Parent |

### 3.3 Best Operating Points

| Pipeline | Best C0 Recall | At Cost | Best Accuracy | At Cost | Best F1 | At Cost |
|----------|----------------|---------|---------------|---------|---------|---------|
| Parent | 86.7% | 60 | 87.7% | 5 | 87.6% | 5 |
| Playground | 78.2% | 2 | 86.9% | 2 | 86.8% | 2 |
| **Hybrid** | **90.0%** | 1 | **89.7%** | 1 | **89.7%** | 1 |

### 3.4 Side-by-Side at Key Cost Values

**Cost = 1 (Baseline):**

| Metric | Parent | Playground | Hybrid |
|--------|--------|------------|--------|
| Accuracy | 86.0% | 84.6% | **89.7%** |
| C0 Recall | 84.0% | 75.6% | **90.0%** |
| C1 Recall | 88.0% | 93.8% | 89.3% |

**Cost = 10:**

| Metric | Parent | Playground | Hybrid |
|--------|--------|------------|--------|
| Accuracy | 86.0% | 74.6% | **88.3%** |
| C0 Recall | 82.0% | 50.7% | **86.0%** |
| C1 Recall | 90.0% | **98.7%** | 90.7% |

**Cost = 100 (Extreme):**

| Metric | Parent | Playground | Hybrid |
|--------|--------|------------|--------|
| Accuracy | 85.3% | 49.9% (collapsed) | **87.3%** |
| C0 Recall | 81.3% | 0% (all-False-Widow) | **83.3%** |
| C1 Recall | 89.3% | 100% | 91.3% |

---

## 4. Comparison with [0][1] Sweep

### 4.1 Collapse Direction

| Cell | What's penalized | Playground collapse direction |
|------|------------------|-------------------------------|
| [0][1] | Black Widow → False Widow (dangerous) | All Black Widow predictions |
| [1][0] | False Widow → Black Widow (safe) | All False Widow predictions |

The Playground's pure CS regularization loss causes it to overfit to whichever direction is penalized, collapsing into single-class predictions at extreme cost values.

### 4.2 Hybrid Stability

The Hybrid loss maintains stable predictions in both directions:

| Cell | Hybrid min accuracy | Hybrid min C0 recall | Playground min accuracy |
|------|---------------------|----------------------|-------------------------|
| [0][1] | 82.7% | 94.0% | 50.1% |
| [1][0] | 83.7% | 77.3% | 49.9% |

### 4.3 Practical Implications

For the spider classification task:
- **[0][1] sweep is more relevant** — we want to catch Black Widows (dangerous) and are willing to accept some False Widows being misclassified as Black Widows
- **[1][0] sweep shows the opposite tradeoff** — penalizing the "safe" error, which may not be as useful practically
- Both sweeps demonstrate that the Hybrid loss provides stability across the full cost range

---

## 5. Why Playground Collapses Differently in [1][0]

In [0][1], M[0][1] = cost_value penalizes predicting class 0 (Black Widow) as class 1 (False Widow). The CS loss pushes the model to avoid this by predicting more class 0.

In [1][0], M[1][0] = cost_value penalizes predicting class 1 (False Widow) as class 0 (Black Widow). The CS loss pushes the model to avoid this by predicting more class 1.

At extreme cost values, the CS term dominates and the model degenerates into always predicting the "safe" class for that direction.

---

## 6. Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `playground/.../train_spiders.py` | Modified | Added `--cost_row`, `--cost_col` args |
| `playground/.../cost_ratio_sweep.py` | Modified | Added `--row`, `--col` args, dynamic titles |
| `playground/.../results/sweep_cost_1_0/` | Created | Playground [1][0] sweep results |
| `results/sweep_cost_1_0_reg/` | Created | Hybrid [1][0] sweep results |
| `results/sweep_comparison_1_0/` | Created | 3-way comparison outputs |
| `docs/2026-01-28_sweep_1_0_comparison.md` | Created | This document |

---

## 7. Output Locations

```
results/sweep_comparison_1_0/
├── comparison_wide.csv
├── comparison_summary.md
├── sweep_comparison.duckdb
└── graphs/
    ├── class_0_recall_comparison.html/.png
    ├── class_1_recall_comparison.html/.png
    ├── accuracy_comparison.html/.png
    ├── f1_comparison.html/.png
    ├── dashboard.html/.png
    └── stability_bar.html/.png
```

---

## 8. Conclusions

1. **Hybrid wins on both [0][1] and [1][0]**: Achieves the best balance of accuracy and recall without collapse.

2. **Playground collapse is symmetric**: High costs cause collapse toward the "safe" class for that penalty direction — all-Black-Widow for [0][1], all-False-Widow for [1][0].

3. **Parent is most conservative**: Stable but doesn't push recall as high as Hybrid.

4. **CS warmup and M-normalization are key**: The Hybrid's stability comes from these mechanisms that prevent the CS term from dominating.

5. **For practical use**: The [0][1] direction (penalizing dangerous misclassifications) is more relevant. Use Hybrid with cost=5-6 for optimal Black Widow detection.
