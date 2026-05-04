# Session Log: HPO Search & Training Pipeline Updates

**Date:** 2026-01-26

## Overview

This session added hyperparameter optimization (HPO) capability to the project, identified optimal hyperparameters for the 2-class spider classification model, extended the training pipeline with new configurable parameters (weight decay, warmup ratio, LR scheduler, layer freezing, early stopping), and ran validation experiments comparing the baseline to the optimized configuration.

---

## 1. New Dependency

```bash
pip install optuna  # installed v4.7.0
```

Optuna provides Bayesian hyperparameter optimization via the Tree-structured Parzen Estimator (TPE) algorithm, with native HuggingFace Trainer integration.

---

## 2. Files Created

### `scripts/hpo_search.py`

Standalone HPO script that reuses existing codebase infrastructure. Key design:

- Loads dataset and pretrained weights **once**, shared across all trials
- `model_init(trial)` creates a fresh model per trial with configurable layer freezing (`num_frozen_stages` 0-3)
- `optuna_hp_space(trial)` defines the search space (see table below)
- Uses `EarlyStoppingCallback(patience=5)` to auto-determine optimal epoch count per trial
- `num_train_epochs=30` ceiling, letting early stopping decide when to stop
- Saves best hyperparameters to `results/hpo_results/best_hyperparameters.json`

**Search space:**

| Parameter | Type | Range |
|---|---|---|
| `learning_rate` | float (log) | 1e-5 to 1e-3 |
| `weight_decay` | float | 0.0 to 0.01 |
| `per_device_train_batch_size` | categorical | [8, 16, 32] |
| `warmup_ratio` | float | 0.0 to 0.2 |
| `lr_scheduler_type` | categorical | ["linear", "cosine"] |
| `num_frozen_stages` | int | 0 to 3 |

**Usage:**
```bash
micromamba activate ml
python scripts/hpo_search.py --config config/2classSpiders.json
```

### `docs/hpo_plan.md`

Copy of the original HPO implementation plan.

---

## 3. Files Modified

### `utils/utils.py` — `ScriptTrainingArguments` dataclass

Added 5 new fields with backward-compatible defaults:

```python
weight_decay: float = field(default=0.0, ...)
warmup_ratio: float = field(default=0.1, ...)
lr_scheduler_type: str = field(default="linear", ...)
num_frozen_stages: int = field(default=0, ...)
early_stopping_patience: int = field(default=0, ...)
```

Existing configs without these fields behave identically to before (defaults match previous hardcoded values).

### `train.py`

Three changes:

1. **Import:** Added `EarlyStoppingCallback` to the transformers import block.

2. **TrainingArguments:** Replaced hardcoded values with configurable ones:
   - `warmup_ratio=0.1` replaced with `script_args.warmup_ratio`
   - Added `weight_decay=script_args.weight_decay`
   - Added `lr_scheduler_type=script_args.lr_scheduler_type`

3. **Layer freezing:** Added logic after weight loading to freeze early ResNet stages based on `num_frozen_stages`:
   - `>= 1`: freezes `model.resnet.embedder` (stem)
   - `>= 2`: also freezes `model.resnet.encoder.stages[0]`
   - `>= 3`: also freezes `model.resnet.encoder.stages[1]`

4. **Early stopping:** Added `EarlyStoppingCallback` to the Trainer when `early_stopping_patience > 0`.

### `config/2classSpiders.json`

Updated from baseline to optimized configuration:

| Field | Before | After |
|---|---|---|
| `learning_rate` | 0.0005 | **0.0003** |
| `num_train_epochs` | 3 | **30** |
| `batch_size` | 16 | **32** |
| `weight_decay` | *(not present)* | **0.01** |
| `warmup_ratio` | *(not present)* | **0.09** |
| `lr_scheduler_type` | *(not present)* | **"linear"** |
| `num_frozen_stages` | *(not present)* | **3** |
| `early_stopping_patience` | *(not present)* | **5** |

---

## 4. HPO Search Results

**20 trials** completed in ~18 minutes on CUDA. 7 ran to early stopping, 13 were pruned by Optuna.

### All Completed Trials

| Trial | Eval Accuracy | LR | Weight Decay | Batch | Warmup | Scheduler | Frozen Stages |
|---|---|---|---|---|---|---|---|
| 0 | 87.78% | 6.42e-4 | 0.009 | 8 | 0.147 | linear | 1 |
| 1 | 89.26% | 2.36e-5 | 0.0003 | 8 | 0.171 | cosine | 1 |
| 2 | 91.48% | 1.55e-4 | 0.0004 | 16 | 0.170 | cosine | 0 |
| 3 | 82.59% | 1.43e-5 | 0.004 | 16 | 0.028 | linear | 2 |
| **4** | **93.70%** | **3.17e-4** | **0.010** | **32** | **0.087** | **linear** | **3** |
| 10 | 92.96% | 8.22e-4 | 0.010 | 32 | 0.121 | linear | 2 |
| 11 | 93.33% | 8.37e-4 | 0.010 | 32 | 0.116 | linear | 2 |

Trials 5-9 and 12-19 were pruned.

### Best Hyperparameters (Trial 4)

```json
{
  "learning_rate": 0.00031675827258233213,
  "weight_decay": 0.009658023670815849,
  "per_device_train_batch_size": 32,
  "warmup_ratio": 0.08656072310315271,
  "lr_scheduler_type": "linear",
  "num_frozen_stages": 3
}
```

Values were rounded for the config file (e.g., lr -> 0.0003, weight_decay -> 0.01).

---

## 5. Training Run Results

### Dataset

- **Source:** `data/2_class_black_widows/`
- **Classes:** Latrodectus_hesperus (1500 images), Steatoda_grossa (1499 images)
- **Split:** Train=2429, Val=270, Test=300

### Run 1: Optimized Config, No Early Stopping (15 epochs)

**Config:** lr=0.0003, batch=32, weight_decay=0.01, warmup=0.09, frozen_stages=3, epochs=15

| Epoch | Val Accuracy | Epoch | Val Accuracy |
|---|---|---|---|
| 1 | 72.22% | 9 | 91.11% |
| 2 | 82.22% | 10 | 91.85% |
| 3 | 87.04% | 11 | 92.22% |
| 4 | 89.63% | **12** | **93.33%** |
| 5 | 91.85% | 13 | 92.59% |
| 6 | 92.59% | 14 | 92.59% |
| 7 | 92.59% | 15 | 92.59% |
| 8 | 92.59% | | |

**Test results** (best model from epoch 12):

| Metric | Value |
|---|---|
| **Test Accuracy** | **89.33%** |
| Test F1 | 89.33% |
| Test Loss | 0.2733 |
| Training Time | 2m 6s |

| | Pred: L. hesperus | Pred: S. grossa |
|---|---|---|
| **True: L. hesperus** | 137 | 13 |
| **True: S. grossa** | 19 | 131 |

Results saved to: `results/resnet_test/resnet_run_01-26_23-44/`

### Run 2: Early Stopping, Patience=3 (stopped at epoch 9)

**Config:** same as Run 1 but with `early_stopping_patience=3`, `num_train_epochs=30`

| Epoch | Val Accuracy |
|---|---|
| 1 | 65.56% |
| 2 | 75.93% |
| 3 | 83.33% |
| 4 | 87.78% |
| 5 | 91.11% |
| **6** | **92.96%** |
| 7 | 92.22% |
| 8 | 90.74% |
| 9 | 92.59% (stopped) |

**Test results** (best model from epoch 6):

| Metric | Value |
|---|---|
| **Test Accuracy** | **87.67%** |
| Test F1 | 87.67% |
| Test Loss | 0.2961 |
| Training Time | 1m 16s |

| | Pred: L. hesperus | Pred: S. grossa |
|---|---|---|
| **True: L. hesperus** | 133 | 17 |
| **True: S. grossa** | 20 | 130 |

Results saved to: `results/resnet_test/resnet_run_01-26_23-55/`

### Comparison Summary

| | Baseline | Run 1 (no ES) | Run 2 (ES p=3) |
|---|---|---|---|
| **Test Accuracy** | 84.67% | **89.33%** | 87.67% |
| Best Val Accuracy | — | 93.33% (ep 12) | 92.96% (ep 6) |
| Epochs Trained | 3 | 15 | 9 (stopped) |
| Training Time | ~30s | 2m 6s | 1m 16s |

**Finding:** Patience=3 was too aggressive — the model's val accuracy oscillates epoch-to-epoch, so 3 consecutive non-improvements is common even while still improving overall. The config has been updated to patience=5.

---

## 6. Current Config State

`config/2classSpiders.json` as of end of session:

```json
{
    "dataset": "local_black_widows",
    "dataset_host": "local_folder",
    "local_folder_path": "data/2_class_black_widows",
    "local_dataset_format": "folder",
    "model": "microsoft/resnet-50",
    "weights": "weights/pytorch_model.bin",
    "learning_rate": 0.0003,
    "num_train_epochs": 30,
    "early_stopping_patience": 5,
    "batch_size": 32,
    "weight_decay": 0.01,
    "warmup_ratio": 0.09,
    "lr_scheduler_type": "linear",
    "num_frozen_stages": 3,
    "num_labels": 2,
    "wandb": "False",
    "push_to_hub": "False",
    "output_dir": "resnet_run",
    "loss_function": "test",
    "cost_matrix": [[0.0, 0.0], [0.0, 0.0]]
}
```

This config has not yet been run with patience=5.
