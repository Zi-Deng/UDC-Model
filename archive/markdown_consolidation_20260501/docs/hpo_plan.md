# Plan: Hyperparameter Optimization for 2-Class Spider Model

## Goal

Find the optimal hyperparameters **and** optimal number of training epochs for the ResNet model fine-tuned on `data/2_class_black_widows/`. The baseline run achieved **84.67% test accuracy** (3 epochs, lr=0.0005, batch_size=16, no weight decay, no layer freezing).

## Approach

Use **Optuna** via HuggingFace Trainer's built-in `hyperparameter_search()` method, combined with `EarlyStoppingCallback` to automatically determine both the best hyperparameters and number of epochs in a single workflow.

**Why Optuna:**
- Native integration with HF Trainer v5.0 (`trainer.hyperparameter_search(backend="optuna")`)
- Tree-structured Parzen Estimator (TPE) algorithm — efficient for small-to-medium search spaces
- Latest version: 4.7.0 (Jan 2026) — actively maintained
- Simpler than Ray Tune for single-GPU, single-node use cases like this one

## New Dependency

```bash
pip install optuna
```

## Implementation

### New file: `scripts/hpo_search.py`

A standalone script that reuses the existing codebase infrastructure. No modifications to existing files needed.

**Structure:**

1. Parse config JSON (reuse `parse_HF_args()`)
2. Load dataset ONCE (shared across all trials)
3. Load pretrained weights ONCE (shared across all trials)
4. Define `model_init(trial)` — fresh model per trial, with optional layer freezing
5. Define `optuna_hp_space(trial)` — search ranges for TrainingArguments params
6. Define `compute_objective(metrics)` — return `eval_accuracy`
7. Create `CustomTrainer` with `model=None` + `model_init` (required for HPO)
8. Add `EarlyStoppingCallback(early_stopping_patience=5)`
9. Call `trainer.hyperparameter_search(backend="optuna", n_trials=20)`
10. Save best hyperparameters to JSON

### Search Space

| Parameter | Type | Range | Rationale |
|---|---|---|---|
| `learning_rate` | float (log scale) | 1e-5 to 1e-3 | Most impactful; current 5e-4 may not be optimal |
| `weight_decay` | float | 0.0 to 0.01 | Regularization — critical for ~2500-sample dataset |
| `per_device_train_batch_size` | categorical | [8, 16, 32] | Affects generalization; smaller = more regularization |
| `warmup_ratio` | float | 0.0 to 0.2 | Training stability; current 0.1 is fixed |
| `lr_scheduler_type` | categorical | ["linear", "cosine"] | Cosine annealing often better for fine-tuning |
| `num_frozen_stages` | int | 0 to 3 | Freezes early ResNet encoder stages. Very impactful for small datasets |

### How Optimal Epochs Is Determined

- `num_train_epochs` set to **30** (high ceiling — lets the model train as long as it improves)
- `EarlyStoppingCallback(early_stopping_patience=5)` — stops training if eval accuracy doesn't improve for 5 consecutive epoch evaluations
- `load_best_model_at_end=True` + `metric_for_best_model="accuracy"` — ensures the best checkpoint is used

### Output

- **Console**: per-trial progress, best hyperparameters, best accuracy
- **File**: `results/hpo_results/best_hyperparameters.json`

## How to Run

```bash
micromamba activate ml
python scripts/hpo_search.py --config config/2classSpiders.json
```
