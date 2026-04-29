# Cost-Sensitive Sweep Comparison: 3-Way Analysis

**Generated**: 2026-01-28 10:10

## 1. Overview

Three cost-sensitive sweep pipelines compared on the binary spider classification task (Black Widow = class 0, False Widow = class 1), all sweeping the M[0][1] cost matrix cell.

**Overlapping cost values** (19): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100

- **Parent (LogitAdj)**: 19 cost values

- **Playground (CE+CS)**: 19 cost values

- **Hybrid (LogitAdj+CS)**: 19 cost values


## 2. Setup Differences

| Property | Parent | Playground | Hybrid |
|---|---|---|---|
| Loss function | CELogitAdjustmentV2 | CostSensitiveRegularizedLoss | CELogitAdjustmentRegularized |
| Framework | HuggingFace Trainer | Custom PyTorch loop | HuggingFace Trainer |
| CS lambda | N/A | 10.0 | 10.0 |
| CS warmup | N/A | N/A | 5 epochs |
| M normalization | No | Implicit (softmax) | Explicit (M/max(M)) |
| Test set size | 300 (150/150) | 449 (225/224) | 300 (150/150) |
| Batch size | 32 | 16 | 32 |
| Weight decay | 0.01 | 0.0 | 0.01 |
| Frozen stages | 3 | 0 | 3 |
| Optimizer | AdamW | Adam | AdamW |
| Max epochs | 30 | 30 | 30 |
| Early stopping | 5 (eval_loss) | 5 (AUC) | 5 (eval_loss) |

## 3. Side-by-Side Results

Abbreviations: **P** = Parent, **PG** = Playground, **H** = Hybrid

### 3.1 Class 0 Recall (Black Widow)

| Cost | P | PG | H | Spread |
|------|---|---|---|--------|
| 1 | 0.8400 | 0.7556 | 0.9000 | 0.1444 |
| 2 | 0.8400 | 0.7822 | 0.8533 | 0.0711 |
| 3 | 0.8000 | 0.6533 | 0.8133 | 0.1600 |
| 4 | 0.8267 | 0.6133 | 0.8000 | 0.2133 |
| 5 | 0.8333 | 0.6133 | 0.7733 | 0.2200 |
| 6 | 0.8533 | 0.6222 | 0.8467 | 0.2311 |
| 7 | 0.8200 | 0.0000 | 0.8600 | 0.8600 |
| 8 | 0.8000 | 0.7022 | 0.8467 | 0.1444 |
| 9 | 0.8067 | 0.6356 | 0.8467 | 0.2111 |
| 10 | 0.8200 | 0.5067 | 0.8600 | 0.3533 |
| 20 | 0.7933 | 0.4667 | 0.8400 | 0.3733 |
| 30 | 0.7867 | 0.4889 | 0.7867 | 0.2978 |
| 40 | 0.7933 | 0.1378 | 0.8533 | 0.7156 |
| 50 | 0.8067 | 0.0667 | 0.8533 | 0.7867 |
| 60 | 0.8667 | 0.2178 | 0.7800 | 0.6489 |
| 70 | 0.8133 | 0.2267 | 0.8200 | 0.5933 |
| 80 | 0.8000 | 0.0044 | 0.8200 | 0.8156 |
| 90 | 0.8267 | 0.0000 | 0.8467 | 0.8467 |
| 100 | 0.8133 | 0.0000 | 0.8333 | 0.8333 |

### 3.2 Accuracy

| Cost | P | PG | H | Spread |
|------|---|---|---|--------|
| 1 | 0.8600 | 0.8463 | 0.8967 | 0.0503 |
| 2 | 0.8633 | 0.8686 | 0.8733 | 0.0100 |
| 3 | 0.8333 | 0.8218 | 0.8467 | 0.0248 |
| 4 | 0.8700 | 0.7996 | 0.8433 | 0.0704 |
| 5 | 0.8767 | 0.7996 | 0.8367 | 0.0771 |
| 6 | 0.8733 | 0.7996 | 0.8667 | 0.0738 |
| 7 | 0.8700 | 0.4989 | 0.8800 | 0.3811 |
| 8 | 0.8533 | 0.8374 | 0.8800 | 0.0426 |
| 9 | 0.8600 | 0.8129 | 0.8800 | 0.0671 |
| 10 | 0.8600 | 0.7461 | 0.8833 | 0.1372 |
| 20 | 0.8500 | 0.7283 | 0.8733 | 0.1450 |
| 30 | 0.8467 | 0.7439 | 0.8600 | 0.1161 |
| 40 | 0.8600 | 0.5679 | 0.8933 | 0.3254 |
| 50 | 0.8400 | 0.5323 | 0.8933 | 0.3610 |
| 60 | 0.8633 | 0.6080 | 0.8600 | 0.2553 |
| 70 | 0.8533 | 0.6125 | 0.8700 | 0.2575 |
| 80 | 0.8567 | 0.5011 | 0.8700 | 0.3689 |
| 90 | 0.8500 | 0.4989 | 0.8767 | 0.3778 |
| 100 | 0.8533 | 0.4989 | 0.8733 | 0.3744 |

### 3.3 F1 Score

| Cost | P | PG | H | Spread |
|------|---|---|---|--------|
| 1 | 0.8599 | 0.8451 | 0.8967 | 0.0516 |
| 2 | 0.8633 | 0.8676 | 0.8733 | 0.0100 |
| 3 | 0.8331 | 0.8167 | 0.8465 | 0.0298 |
| 4 | 0.8698 | 0.7925 | 0.8430 | 0.0773 |
| 5 | 0.8764 | 0.7925 | 0.8360 | 0.0840 |
| 6 | 0.8733 | 0.7932 | 0.8666 | 0.0801 |
| 7 | 0.8697 | 0.3328 | 0.8800 | 0.5471 |
| 8 | 0.8529 | 0.8345 | 0.8799 | 0.0454 |
| 9 | 0.8596 | 0.8069 | 0.8799 | 0.0729 |
| 10 | 0.8598 | 0.7308 | 0.8833 | 0.1525 |
| 20 | 0.8495 | 0.7085 | 0.8732 | 0.1647 |
| 30 | 0.8461 | 0.7262 | 0.8592 | 0.1330 |
| 40 | 0.8594 | 0.4700 | 0.8932 | 0.4232 |
| 50 | 0.8398 | 0.4029 | 0.8932 | 0.4902 |
| 60 | 0.8633 | 0.5378 | 0.8591 | 0.3255 |
| 70 | 0.8531 | 0.5449 | 0.8697 | 0.3248 |
| 80 | 0.8562 | 0.3378 | 0.8697 | 0.5319 |
| 90 | 0.8499 | 0.3328 | 0.8766 | 0.5437 |
| 100 | 0.8531 | 0.3328 | 0.8731 | 0.5403 |

### 3.4 Class 1 Recall (False Widow)

| Cost | P | PG | H | Spread |
|------|---|---|---|--------|
| 1 | 0.8800 | 0.9375 | 0.8933 | 0.0575 |
| 2 | 0.8867 | 0.9554 | 0.8933 | 0.0687 |
| 3 | 0.8667 | 0.9911 | 0.8800 | 0.1244 |
| 4 | 0.9133 | 0.9866 | 0.8867 | 0.0999 |
| 5 | 0.9200 | 0.9866 | 0.9000 | 0.0866 |
| 6 | 0.8933 | 0.9777 | 0.8867 | 0.0910 |
| 7 | 0.9200 | 1.0000 | 0.9000 | 0.1000 |
| 8 | 0.9067 | 0.9732 | 0.9133 | 0.0665 |
| 9 | 0.9133 | 0.9911 | 0.9133 | 0.0777 |
| 10 | 0.9000 | 0.9866 | 0.9067 | 0.0866 |
| 20 | 0.9067 | 0.9911 | 0.9067 | 0.0844 |
| 30 | 0.9067 | 1.0000 | 0.9333 | 0.0933 |
| 40 | 0.9267 | 1.0000 | 0.9333 | 0.0733 |
| 50 | 0.8733 | 1.0000 | 0.9333 | 0.1267 |
| 60 | 0.8600 | 1.0000 | 0.9400 | 0.1400 |
| 70 | 0.8933 | 1.0000 | 0.9200 | 0.1067 |
| 80 | 0.9133 | 1.0000 | 0.9200 | 0.0867 |
| 90 | 0.8733 | 1.0000 | 0.9067 | 0.1267 |
| 100 | 0.8933 | 1.0000 | 0.9133 | 0.1067 |

## 4. Key Findings

### 4.1 Baseline Comparison (cost=1)

| Metric | Parent | Playground | Hybrid |
|---|---|---|---|
| accuracy | 0.8600 | 0.8463 | 0.8967 |
| class_0_recall | 0.8400 | 0.7556 | 0.9000 |
| class_1_recall | 0.8800 | 0.9375 | 0.8933 |
| f1_score | 0.8599 | 0.8451 | 0.8967 |

### 4.2 Best Class 0 Recall per Pipeline

| Pipeline | Best C0 Recall | At Cost | Accuracy at that Cost |
|---|---|---|---|
| Parent (LogitAdj) | 0.8667 | 60 | 0.8633 |
| Playground (CE+CS) | 0.7822 | 2 | 0.8686 |
| Hybrid (LogitAdj+CS) | 0.9000 | 1 | 0.8967 |

### 4.3 Collapse Detection

A pipeline is considered collapsed if accuracy falls below 55% at any cost value.

- **Parent (LogitAdj)**: No collapse (min accuracy: 83.3%)
- **Playground (CE+CS)**: COLLAPSED at cost(s) 7, 50, 80, 90, 100 (min accuracy: 49.9%)
- **Hybrid (LogitAdj+CS)**: No collapse (min accuracy: 83.7%)

## 5. Stability Analysis

Spread = max(metric) - min(metric) across all cost values. Lower is more stable.

| Metric | Parent | Playground | Hybrid | Most Stable |
|---|---|---|---|---|
| class_0_recall | 0.0800 | 0.7822 | 0.1267 | Parent (LogitAdj) |
| accuracy | 0.0433 | 0.3697 | 0.0600 | Parent (LogitAdj) |
| f1_score | 0.0433 | 0.5348 | 0.0607 | Parent (LogitAdj) |
| class_1_recall | 0.0667 | 0.0625 | 0.0600 | Hybrid (LogitAdj+CS) |
| class_0_precision | 0.0582 | 1.0000 | 0.0571 | Hybrid (LogitAdj+CS) |
| class_1_precision | 0.0562 | 0.3148 | 0.1005 | Parent (LogitAdj) |

## 6. Confusion Matrix at Selected Cost Values

Note: Parent/Hybrid test set = 300 samples (150/150). Playground = 449 (225/224).

### cost = 1

**Parent (LogitAdj)**:
```
            Pred BW  Pred FW
True BW        126       24
True FW         18      132
```

**Playground (CE+CS)**:
```
            Pred BW  Pred FW
True BW        170       55
True FW         14      210
```

**Hybrid (LogitAdj+CS)**:
```
            Pred BW  Pred FW
True BW        135       15
True FW         16      134
```


### cost = 10

**Parent (LogitAdj)**:
```
            Pred BW  Pred FW
True BW        123       27
True FW         15      135
```

**Playground (CE+CS)**:
```
            Pred BW  Pred FW
True BW        114      111
True FW          3      221
```

**Hybrid (LogitAdj+CS)**:
```
            Pred BW  Pred FW
True BW        129       21
True FW         14      136
```


### cost = 50

**Parent (LogitAdj)**:
```
            Pred BW  Pred FW
True BW        121       29
True FW         19      131
```

**Playground (CE+CS)**:
```
            Pred BW  Pred FW
True BW         15      210
True FW          0      224
```

**Hybrid (LogitAdj+CS)**:
```
            Pred BW  Pred FW
True BW        128       22
True FW         10      140
```


### cost = 100

**Parent (LogitAdj)**:
```
            Pred BW  Pred FW
True BW        122       28
True FW         16      134
```

**Playground (CE+CS)**:
```
            Pred BW  Pred FW
True BW          0      225
True FW          0      224
```

**Hybrid (LogitAdj+CS)**:
```
            Pred BW  Pred FW
True BW        125       25
True FW         13      137
```


## 7. Best Operating Points

| Pipeline | Best C0 Recall | At Cost | Best Accuracy | At Cost | Best F1 | At Cost |
|---|---|---|---|---|---|---|
| Parent (LogitAdj) | 0.8667 | 60 | 0.8767 | 5 | 0.8764 | 5 |
| Playground (CE+CS) | 0.7822 | 2 | 0.8686 | 2 | 0.8676 | 2 |
| Hybrid (LogitAdj+CS) | 0.9000 | 1 | 0.8967 | 1 | 0.8967 | 1 |

## 8. Conclusions

1. **Most stable pipeline**: Hybrid (LogitAdj+CS) (minimum accuracy 83.7% across all cost values).
2. **Highest Class 0 Recall**: Hybrid (LogitAdj+CS) achieves 90.0%.
3. **Collapse**: 1 of 3 pipelines collapse at high cost values.

## 9. Data Sources

| Dataset | Path |
|---|---|
| Parent sweep CSV | `results/sweep_cost_0_1/sweep_results.csv` |
| Playground sweep CSV | `playground/cost_sensitive_loss_classification/results/sweep_cost_ratio/sweep_results.csv` |
| Hybrid sweep CSV | `results/sweep_cost_0_1_reg/sweep_results.csv` |
| Comparison output | `results/sweep_comparison_1_0/` |
