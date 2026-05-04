# Cost-Sensitive Sweep Comparison: 3-Way Analysis

**Generated**: 2026-01-28 07:53

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
| 1 | 0.9467 | 0.9689 | 0.9400 | 0.0289 |
| 2 | 0.9200 | 0.9600 | 0.9467 | 0.0400 |
| 3 | 0.9133 | 0.9422 | 0.9667 | 0.0533 |
| 4 | 0.9200 | 0.9644 | 0.9600 | 0.0444 |
| 5 | 0.9067 | 0.9556 | 0.9533 | 0.0489 |
| 6 | 0.9067 | 0.9778 | 0.9800 | 0.0733 |
| 7 | 0.9333 | 0.9822 | 0.9533 | 0.0489 |
| 8 | 0.9333 | 0.9600 | 0.9467 | 0.0267 |
| 9 | 0.9067 | 0.9867 | 0.9667 | 0.0800 |
| 10 | 0.8933 | 0.9644 | 0.9733 | 0.0800 |
| 20 | 0.9133 | 0.9644 | 0.9600 | 0.0511 |
| 30 | 0.9267 | 0.9911 | 0.9400 | 0.0644 |
| 40 | 0.9400 | 0.9911 | 0.9667 | 0.0511 |
| 50 | 0.9333 | 1.0000 | 0.9467 | 0.0667 |
| 60 | 0.9200 | 0.9956 | 0.9667 | 0.0756 |
| 70 | 0.9267 | 0.9911 | 0.9200 | 0.0711 |
| 80 | 0.9400 | 1.0000 | 0.9467 | 0.0600 |
| 90 | 0.9067 | 0.9956 | 0.9467 | 0.0889 |
| 100 | 0.9000 | 1.0000 | 0.9533 | 0.1000 |

### 3.2 Accuracy

| Cost | P | PG | H | Spread |
|------|---|---|---|--------|
| 1 | 0.8933 | 0.8886 | 0.8500 | 0.0433 |
| 2 | 0.8867 | 0.8552 | 0.8767 | 0.0314 |
| 3 | 0.8767 | 0.8530 | 0.8467 | 0.0300 |
| 4 | 0.8867 | 0.8174 | 0.8867 | 0.0693 |
| 5 | 0.8833 | 0.6793 | 0.8967 | 0.2174 |
| 6 | 0.8900 | 0.8218 | 0.8867 | 0.0682 |
| 7 | 0.8533 | 0.7973 | 0.8700 | 0.0727 |
| 8 | 0.8567 | 0.8463 | 0.8400 | 0.0167 |
| 9 | 0.8600 | 0.7617 | 0.8733 | 0.1116 |
| 10 | 0.8533 | 0.8040 | 0.8667 | 0.0627 |
| 20 | 0.8667 | 0.7639 | 0.8333 | 0.1027 |
| 30 | 0.8767 | 0.7327 | 0.8333 | 0.1439 |
| 40 | 0.8500 | 0.6659 | 0.8867 | 0.2207 |
| 50 | 0.8533 | 0.6058 | 0.8267 | 0.2475 |
| 60 | 0.8567 | 0.6748 | 0.8700 | 0.1952 |
| 70 | 0.8733 | 0.6837 | 0.8667 | 0.1896 |
| 80 | 0.8367 | 0.5011 | 0.8667 | 0.3656 |
| 90 | 0.8233 | 0.5434 | 0.8533 | 0.3099 |
| 100 | 0.8700 | 0.5011 | 0.8633 | 0.3689 |

### 3.3 F1 Score

| Cost | P | PG | H | Spread |
|------|---|---|---|--------|
| 1 | 0.8930 | 0.8879 | 0.8488 | 0.0443 |
| 2 | 0.8865 | 0.8536 | 0.8761 | 0.0330 |
| 3 | 0.8765 | 0.8518 | 0.8444 | 0.0321 |
| 4 | 0.8865 | 0.8132 | 0.8861 | 0.0733 |
| 5 | 0.8833 | 0.6523 | 0.8963 | 0.2440 |
| 6 | 0.8900 | 0.8173 | 0.8857 | 0.0727 |
| 7 | 0.8524 | 0.7900 | 0.8691 | 0.0791 |
| 8 | 0.8558 | 0.8442 | 0.8382 | 0.0177 |
| 9 | 0.8597 | 0.7487 | 0.8722 | 0.1235 |
| 10 | 0.8531 | 0.7987 | 0.8651 | 0.0664 |
| 20 | 0.8664 | 0.7538 | 0.8306 | 0.1126 |
| 30 | 0.8764 | 0.7133 | 0.8314 | 0.1631 |
| 40 | 0.8488 | 0.6258 | 0.8859 | 0.2601 |
| 50 | 0.8524 | 0.5323 | 0.8241 | 0.3201 |
| 60 | 0.8561 | 0.6370 | 0.8688 | 0.2318 |
| 70 | 0.8730 | 0.6502 | 0.8663 | 0.2228 |
| 80 | 0.8349 | 0.3338 | 0.8658 | 0.5320 |
| 90 | 0.8221 | 0.4247 | 0.8520 | 0.4274 |
| 100 | 0.8699 | 0.3338 | 0.8622 | 0.5361 |

### 3.4 Class 1 Recall (False Widow)

| Cost | P | PG | H | Spread |
|------|---|---|---|--------|
| 1 | 0.8400 | 0.8080 | 0.7600 | 0.0800 |
| 2 | 0.8533 | 0.7500 | 0.8067 | 0.1033 |
| 3 | 0.8400 | 0.7634 | 0.7267 | 0.1133 |
| 4 | 0.8533 | 0.6696 | 0.8133 | 0.1837 |
| 5 | 0.8600 | 0.4018 | 0.8400 | 0.4582 |
| 6 | 0.8733 | 0.6652 | 0.7933 | 0.2082 |
| 7 | 0.7733 | 0.6116 | 0.7867 | 0.1751 |
| 8 | 0.7800 | 0.7321 | 0.7333 | 0.0479 |
| 9 | 0.8133 | 0.5357 | 0.7800 | 0.2776 |
| 10 | 0.8133 | 0.6429 | 0.7600 | 0.1705 |
| 20 | 0.8200 | 0.5625 | 0.7067 | 0.2575 |
| 30 | 0.8267 | 0.4732 | 0.7267 | 0.3535 |
| 40 | 0.7600 | 0.3393 | 0.8067 | 0.4674 |
| 50 | 0.7733 | 0.2098 | 0.7067 | 0.5635 |
| 60 | 0.7933 | 0.3527 | 0.7733 | 0.4407 |
| 70 | 0.8200 | 0.3750 | 0.8133 | 0.4450 |
| 80 | 0.7333 | 0.0000 | 0.7867 | 0.7867 |
| 90 | 0.7400 | 0.0893 | 0.7600 | 0.6707 |
| 100 | 0.8400 | 0.0000 | 0.7733 | 0.8400 |

## 4. Key Findings

### 4.1 Baseline Comparison (cost=1)

| Metric | Parent | Playground | Hybrid |
|---|---|---|---|
| accuracy | 0.8933 | 0.8886 | 0.8500 |
| class_0_recall | 0.9467 | 0.9689 | 0.9400 |
| class_1_recall | 0.8400 | 0.8080 | 0.7600 |
| f1_score | 0.8930 | 0.8879 | 0.8488 |

### 4.2 Best Class 0 Recall per Pipeline

| Pipeline | Best C0 Recall | At Cost | Accuracy at that Cost |
|---|---|---|---|
| Parent (LogitAdj) | 0.9467 | 1 | 0.8933 |
| Playground (CE+CS) | 1.0000 | 50 | 0.6058 |
| Hybrid (LogitAdj+CS) | 0.9800 | 6 | 0.8867 |

### 4.3 Collapse Detection

A pipeline is considered collapsed if accuracy falls below 55% at any cost value.

- **Parent (LogitAdj)**: No collapse (min accuracy: 82.3%)
- **Playground (CE+CS)**: COLLAPSED at cost(s) 80, 90, 100 (min accuracy: 50.1%)
- **Hybrid (LogitAdj+CS)**: No collapse (min accuracy: 82.7%)

## 5. Stability Analysis

Spread = max(metric) - min(metric) across all cost values. Lower is more stable.

| Metric | Parent | Playground | Hybrid | Most Stable |
|---|---|---|---|---|
| class_0_recall | 0.0533 | 0.0578 | 0.0600 | Parent (LogitAdj) |
| accuracy | 0.0700 | 0.3875 | 0.0700 | Parent (LogitAdj) |
| f1_score | 0.0709 | 0.5541 | 0.0722 | Parent (LogitAdj) |
| class_1_recall | 0.1400 | 0.8080 | 0.1333 | Hybrid (LogitAdj+CS) |
| class_0_precision | 0.1003 | 0.3341 | 0.0928 | Hybrid (LogitAdj+CS) |
| class_1_precision | 0.0562 | 1.0000 | 0.0650 | Parent (LogitAdj) |

## 6. Confusion Matrix at Selected Cost Values

Note: Parent/Hybrid test set = 300 samples (150/150). Playground = 449 (225/224).

### cost = 1

**Parent (LogitAdj)**:
```
            Pred BW  Pred FW
True BW        142        8
True FW         24      126
```

**Playground (CE+CS)**:
```
            Pred BW  Pred FW
True BW        218        7
True FW         43      181
```

**Hybrid (LogitAdj+CS)**:
```
            Pred BW  Pred FW
True BW        141        9
True FW         36      114
```


### cost = 10

**Parent (LogitAdj)**:
```
            Pred BW  Pred FW
True BW        134       16
True FW         28      122
```

**Playground (CE+CS)**:
```
            Pred BW  Pred FW
True BW        217        8
True FW         80      144
```

**Hybrid (LogitAdj+CS)**:
```
            Pred BW  Pred FW
True BW        146        4
True FW         36      114
```


### cost = 50

**Parent (LogitAdj)**:
```
            Pred BW  Pred FW
True BW        140       10
True FW         34      116
```

**Playground (CE+CS)**:
```
            Pred BW  Pred FW
True BW        225        0
True FW        177       47
```

**Hybrid (LogitAdj+CS)**:
```
            Pred BW  Pred FW
True BW        142        8
True FW         44      106
```


### cost = 100

**Parent (LogitAdj)**:
```
            Pred BW  Pred FW
True BW        135       15
True FW         24      126
```

**Playground (CE+CS)**:
```
            Pred BW  Pred FW
True BW        225        0
True FW        224        0
```

**Hybrid (LogitAdj+CS)**:
```
            Pred BW  Pred FW
True BW        143        7
True FW         34      116
```


## 7. Best Operating Points

| Pipeline | Best C0 Recall | At Cost | Best Accuracy | At Cost | Best F1 | At Cost |
|---|---|---|---|---|---|---|
| Parent (LogitAdj) | 0.9467 | 1 | 0.8933 | 1 | 0.8930 | 1 |
| Playground (CE+CS) | 1.0000 | 50 | 0.8886 | 1 | 0.8879 | 1 |
| Hybrid (LogitAdj+CS) | 0.9800 | 6 | 0.8967 | 5 | 0.8963 | 5 |

## 8. Conclusions

1. **Most stable pipeline**: Hybrid (LogitAdj+CS) (minimum accuracy 82.7% across all cost values).
2. **Highest Class 0 Recall**: Playground (CE+CS) achieves 100.0%.
3. **Collapse**: 1 of 3 pipelines collapse at high cost values.

## 9. Data Sources

| Dataset | Path |
|---|---|
| Parent sweep CSV | `results/sweep_cost_0_1/sweep_results.csv` |
| Playground sweep CSV | `playground/cost_sensitive_loss_classification/results/sweep_cost_ratio/sweep_results.csv` |
| Hybrid sweep CSV | `results/sweep_cost_0_1_reg/sweep_results.csv` |
| Comparison output | `results/sweep_comparison/` |
