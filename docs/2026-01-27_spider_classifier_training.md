# Session Log: Spider Classifier with Cost-Sensitive Loss

**Date:** 2026-01-27

## Overview

Implemented `playground/cost_sensitive_loss_classification/train_spiders.py`, a binary spider classifier (Black Widow vs False Widow) using the cost-sensitive regularized loss from the MICCAI 2020 paper. The script was verified with a smoke test (3 epochs), then a full training run was executed: ResNet-50 with ImageNet pretraining, early-stopped at epoch 22, achieving **0.9644 AUC** and **0.7617 quadratic kappa** on the validation set.

---

## 1. What Was Built

**New file:** `playground/cost_sensitive_loss_classification/train_spiders.py` (~300 lines)

### Components

| Function / Section | Purpose |
|---|---|
| `prepare_spider_csvs()` | Scans `data/2_class_black_widows/` folder structure, writes stratified train/val CSVs using Polars |
| `eval_predictions_binary()` | Binary evaluation (kappa, AUC, balanced accuracy) -- replaces `eval_predictions_multi()` which hardcodes 5 classes and crashes on binary AUC |
| `print_cm_binary()` | Pretty-prints 2x2 confusion matrix with class display names |
| `build_cost_matrix()` | Returns asymmetric 2x2 cost matrix as `torch.Tensor` |
| `get_spider_criterion()` | Builds `CostSensitiveRegularizedLoss` and overrides `self.M` with custom matrix |
| `run_one_epoch()` | Forward/backward pass loop (adapted from `train.py`) |
| `train_model()` | Training loop with early stopping, LR decay at 3/4 patience, EWMA smoothing |
| `parse_args()` | argparse CLI with spider-appropriate defaults |

### Reused from existing codebase

| Module | What |
|---|---|
| `utils/losses.py` | `CostSensitiveRegularizedLoss` (with `self.M` override) |
| `utils/get_loaders.py` | `DRDataset` (reads CSV with `image_id`, `dr` columns) |
| `models/get_model.py` | `get_arch()` for ResNet-50 with pretrained weights |
| `utils/model_saving_loading.py` | `write_model()` for checkpoint saving |
| `utils/reproducibility.py` | `set_seeds()` |
| `utils/evaluation.py` | `ewma()` for smoothing validation metrics |

---

## 2. Key Design Decisions

### Class ordering (hardcoded, not `os.listdir`)
- Class 0: `Latrodectus_hesperus` (Black Widow -- dangerous)
- Class 1: `Steatoda_grossa` (False Widow -- harmless)

### Asymmetric cost matrix
```
M = [[0.0, cost_ratio],   # True=Black Widow predicted as harmless -> penalty
     [0.0, 0.0       ]]   # True=False Widow predicted as dangerous -> no CS penalty
```
Only `M[0][1]` is non-zero. The base CE loss still penalizes all errors; the CS term adds extra penalty only for the dangerous misclassification direction.

### Binary AUC fix
`roc_auc_score` with `multi_class='ovo'` crashes for 2-class probability arrays. Fixed by passing `y_proba[:, 1]` (1D positive-class probabilities).

### Image handling
- 256x256 (not 512x512) -- natural photos, not medical scans
- RGB conversion via `tr.Lambda(lambda img: img.convert("RGB"))` -- 7 PNG files in dataset are potentially RGBA
- More aggressive ColorJitter than the DR pipeline (brightness/contrast/saturation=0.25, hue=0.05)

### Data preparation
Uses Polars (per project convention) with stratified 85/15 split. Dataset: 1500 Black Widow + 1499 False Widow images.

---

## 3. Training Configuration (Full Run)

| Parameter | Value |
|---|---|
| `model_name` | `resnet50` |
| `pretrained` | `true` (ImageNet) |
| `base_loss` | `ce` (cross-entropy) |
| `lambd` | `10.0` |
| `cost_ratio` | `1.0` |
| `lr` | `0.0003` |
| `batch_size` | `16` |
| `optimizer` | `adam` |
| `n_epochs` | `30` |
| `patience` | `5` |
| `decay_f` | `0.1` |
| `metric` | `auc` |
| `img_size` | `256` |
| `val_fraction` | `0.15` |
| `seed` | `42` |

Data split: 1275/225 Black Widow, 1275/224 False Widow (train/val).

---

## 4. Results

### Best validation metrics (EWMA-smoothed)

| Metric | Value |
|---|---|
| **AUC** | **0.9644** |
| **Quadratic Kappa** | **0.7617** |

### Training progression

| Epoch | Train Loss | Val Loss | Train AUC | Val AUC | Train Kappa | Val Kappa | Event |
|---|---|---|---|---|---|---|---|
| 1 | 1.3000 | 1.1526 | 0.8350 | 0.8872 | 0.2886 | 0.1610 | |
| 4 | 0.9432 | 0.8958 | 0.9101 | 0.9247 | 0.5247 | 0.4470 | |
| 8 | 0.6632 | 0.6927 | 0.9563 | 0.9559 | 0.6573 | 0.6479 | LR decay 3e-4 -> 3e-5 |
| 13 | 0.4725 | 0.6630 | 0.9728 | 0.9642 | 0.7741 | 0.7416 | |
| 18 | 0.3645 | 0.6524 | 0.9845 | 0.9661 | 0.8361 | 0.7728 | Best checkpoint saved |
| 22 | 0.2967 | 0.6791 | 0.9895 | 0.9632 | 0.8627 | 0.7505 | LR decay 3e-5 -> 3e-6 |
| 23 | 0.3104 | 0.6967 | 0.9879 | 0.9634 | 0.8557 | 0.7638 | Early stopping triggered |

### Observations
- Model converged well with ImageNet transfer learning; AUC exceeded 0.88 after just 1 epoch.
- LR decay at epoch 8 (3/4 of patience) produced a clear jump in both train and val metrics.
- Mild overfitting visible after epoch 13: train loss continued decreasing while val loss plateaued around 0.65-0.70. Early stopping handled this appropriately.
- Best checkpoint at epoch 18 (smoothed val AUC = 0.9644).

---

## 5. Artifacts Produced

| Artifact | Path |
|---|---|
| Training script | `playground/cost_sensitive_loss_classification/train_spiders.py` |
| Training CSV | `playground/cost_sensitive_loss_classification/data/train_spiders.csv` |
| Validation CSV | `playground/cost_sensitive_loss_classification/data/val_spiders.csv` |
| Model checkpoint | `playground/cost_sensitive_loss_classification/experiments/spider_full/model_checkpoint.pth` (283 MB) |
| Config | `playground/cost_sensitive_loss_classification/experiments/spider_full/config.json` |
| Metrics | `playground/cost_sensitive_loss_classification/experiments/spider_full/val_metrics.txt` |

---

## 6. How to Reproduce

```bash
cd <repo-root>/playground/cost_sensitive_loss_classification
micromamba run -n ml python train_spiders.py --save_path spider_full
```

To vary the cost-sensitive penalty:
```bash
micromamba run -n ml python train_spiders.py --cost_ratio 5.0 --lambd 20 --save_path spider_high_penalty
```
