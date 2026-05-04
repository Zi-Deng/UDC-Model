# Training Pipeline

## Parent Pipeline

The parent training entry point is `scripts/train.py`.

High-level flow:

1. `parse_HF_args()` in `utils/utils.py` reads `--config <path>`.
2. JSON keys are parsed into `ScriptTrainingArguments`.
3. Dataset loading dispatches by `dataset_host`.
4. `CustomImageProcessor.from_pretrained(script_args.model)` creates preprocessing transforms.
5. Pretrained weights are loaded from `script_args.weights` with `torch.load(..., map_location="cpu")`.
6. `model_type` selects either custom ResNet or custom ConvNeXt.
7. Classifier-head weights are excluded from loaded pretrained weights.
8. Optional early ResNet stages are frozen via `num_frozen_stages`.
9. HuggingFace `TrainingArguments` is built.
10. `CustomTrainer` injects a selected custom loss through `LossFunctions.loss_function`.
11. Training runs against the train split and validates each epoch.
12. A second `CustomTrainer` evaluates on the test split.
13. `perform_comprehensive_evaluation()` writes JSON/TXT metrics, run config, and confusion matrix visualizations.

## Dataset Sources

`ScriptTrainingArguments.dataset_host` supports:

- `huggingface`: `preprocess_hf_dataset(dataset, model)`
- `kaggle`: `preprocess_kg_dataset(dataset, local_dataset_name, model)`
- `local_folder`: one of:
  - `local_dataset_format == "folder"`: class subdirectories with images.
  - `local_dataset_format == "csv"`: `train_dataset.csv` and `test_dataset.csv` with `image_path` and `binary_label`.

Local folder image extensions recognized by parent code:

```text
.jpg, .jpeg, .png, .bmp, .tiff, .tif
```

Folder-format local splits are stratified:

- test fraction: 0.1
- validation fraction: 0.1 of remaining train+val
- random state: 42

CSV-format local datasets split only the training CSV into train/val and use the provided test CSV for test.

## Models

`model/ResNet.py` implements ResNet variants with `ResNetConfig`, `ResNetModel`, and `ResNetForImageClassification`. The default parent config creates ResNet-50 shape via depths `[3, 4, 6, 3]` and bottleneck blocks.

`model/convnext.py` implements ConvNeXt with patch embeddings, stages, custom layer norm, and `ConvNextForImageClassification`.

`model/__init__.py` provides local replacements for selected HuggingFace base classes/output containers:

- `CustomConfig`
- `CustomPreTrainedModel`
- `BaseModelOutputWithNoAttention`
- `BaseModelOutputWithPoolingAndNoAttention`
- `ImageClassifierOutputWithNoAttention`

The model forward methods can compute standard losses if labels are supplied, but parent training normally ignores that and computes loss in `CustomTrainer.compute_loss`.

## Image Processor

`utils/image_processor.py` defines `CustomImageProcessor`, a local replacement for `AutoImageProcessor`.

It auto-detects model family from the model name:

- `resnet`
- `vit` / `vision`
- `convnext`
- `efficientnet`
- `swin`

Unknown names default to ResNet normalization/crop settings.

Training transform defaults:

- random resized crop
- random horizontal flip
- no color jitter unless requested
- tensor conversion and normalization

Validation/test transform defaults:

- resize
- center crop
- tensor conversion and normalization

## Loss Dispatch

`utils/loss_functions.py` dispatches by config string:

| Config value | Method | Notes |
|---|---|---|
| `cross_entropy` | `cross_entropy()` | custom CE implementation from logits |
| `seesaw` | `seesaw_loss()` | rebalancing-style loss |
| `cost_matrix_cross_entropy` | `CELossLTV1()` | dynamic-alpha cost-matrix CE |
| `logit_adjustment` | `CELogitAdjustmentV2()` | adjusts predicted max-class logit for misclassified examples |
| `test` | `CELogitAdjustmentV2()` | alias |
| `logit_adjustment_regularized` | `CELogitAdjustmentRegularized()` | logit adjustment plus normalized expected-cost regularizer |

Cost-matrix conventions in parent losses are not perfectly uniform:

- `CELogitAdjustmentV2` and `CELogitAdjustmentRegularized` use `cost_matrix[targets, pred_classes]`.
- `CELossLTV1` uses `cost_matrix[predicted_classes, targets]`.

Be careful when interpreting row/column semantics across loss implementations and sweep scripts.

## Regularized Hybrid Loss

`CELogitAdjustmentRegularized` combines:

- Logit adjustment like V2.
- A cost-sensitive regularization term: normalized cost matrix row for each target dotted with softmax probabilities.
- Optional warmup controlled by `cs_warmup_epochs`.
- Strength controlled by `cs_lambda`.

`scripts/train_reg.py` is a thin wrapper intended for this loss. It validates that `cost_matrix` exists and warns if `loss_function` is not `logit_adjustment_regularized`.

## Metrics And Outputs

Evaluation computes:

- accuracy
- macro F1
- balanced accuracy
- Cohen kappa
- AUC
- class 0 recall
- expected cost from the cost matrix
- class 0 prevalence
- per-class precision, recall, F1, FPR, FNR, support
- confusion matrix

Parent results are always written under:

```text
results/resnet_test/<output_dir>_<MM-DD>_<HH-MM>/
```

In sweep mode:

```text
results/resnet_test/<output_dir>__modifier<cost>_true<row>_predict<col>_<MM-DD>_<HH-MM>/
```

Each run normally includes:

- `run_configuration_<timestamp>.json`
- `metrics_<timestamp>_<loss_function>.json`
- `metrics_<timestamp>_<loss_function>.txt`
- `confusion_matrix_<timestamp>_<loss_function>.png`
- `confusion_matrix_<timestamp>_<loss_function>.pdf`
- `confusion_matrix_detailed_<timestamp>_<loss_function>.png`

