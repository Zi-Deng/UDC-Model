# Data, Configs, And Artifacts

## Config Files

`config/modelConfig.json`

- HuggingFace dataset: `zkdeng/spiderTraining5-100`
- model: `microsoft/resnet-50`
- weights: `weights/pytorch_model.bin`
- epochs: 2
- batch size: 16
- labels: 5
- loss: `logit_adjustment`
- cost matrix: 5x5 with one nonzero cell `[0][3] = 10.0`

`config/2classSpiders.json`

- local folder dataset: `data/2_class_black_widows`
- local format: `folder`
- model: `microsoft/resnet-50`
- weights: `weights/pytorch_model.bin`
- learning rate: `0.0003`
- epochs: 30
- early stopping patience: 5
- batch size: 32
- weight decay: 0.01
- warmup ratio: 0.09
- scheduler: `linear`
- frozen ResNet stages: 3
- labels: 2
- loss: `logit_adjustment`
- cost matrix: all zeros
- legacy-compatible config; canonical NICME equivalent is `config/nicme_2class_spiders.json`

`config/2classSpiders_reg.json`

- same local dataset and tuned training parameters as `2classSpiders.json`
- output dir: `resnet_reg`
- loss: `logit_adjustment_regularized`
- cost matrix: `[[0.0, 1.0], [0.0, 0.0]]`
- `cs_lambda`: 10.0
- `cs_warmup_epochs`: 5
- legacy-compatible config; canonical NICME equivalent is `config/nicme_2class_spiders_regularized.json`

`config/sweep_2class_bw_cost.json`

- bash sweep config
- cost range: 0.0 to 10.0 by 0.5
- cell: row 0, col 1
- base config: `config/2classSpiders.json`
- experiment output dir: `results/sweep_bw_misclass_cost_0_1`
- legacy-compatible config; canonical NICME equivalent is `config/nicme_sweep_2class_bw_cost.json`

Canonical NICME configs use JSON booleans for `wandb` and `push_to_hub`; legacy configs with string booleans are normalized during parsing.

`config/image_processor_configs.json`

- reference image normalization/crop configs for ResNet, ConvNeXt, ViT, EfficientNet, Swin, and custom high-res.
- This file is not directly loaded by `CustomImageProcessor.from_pretrained()` unless a caller uses `from_config`; defaults are also embedded in `utils/image_processor.py`.

## Local Data

Local dataset counts found during inspection:

```text
data/2_class_black_widows/Latrodectus_hesperus: 1500 files
data/2_class_black_widows/Steatoda_grossa: 1499 files
data/3_class_black_widows/Latrodectus_hesperus: 1500 files
data/3_class_black_widows/Steatoda_grossa: 1500 files
data/3_class_black_widows/Steatoda_nobilis: 1500 files
```

Parent folder preprocessing ignores hidden class folders, and image collection filters by extension.

## Weights

The local pretrained weights file exists at:

```text
weights/pytorch_model.bin
```

Inspection reported size:

```text
102,567,489 bytes
```

Parent training requires this file for the default configs. Classifier weights are filtered out before loading into the custom model:

```python
filtered_weights = {k: v for k, v in pretrained_weights.items() if "classifier" not in k}
```

## Generated Artifacts

Large generated directories found during inspection:

```text
checkpoints/: 19G
results/: 284M
data/: 732M
playground/cost_sensitive_loss_classification/: 5.4G
```

`checkpoints/` contains HuggingFace Trainer checkpoint folders for parent runs and HPO.

`results/` contains parent metrics, sweep CSVs/DuckDB databases, plots, comparison reports, and HPO results.

`playground/cost_sensitive_loss_classification/experiments/` contains saved playground models and metrics for spider sweeps.

Do not delete or rewrite generated artifacts unless the user explicitly asks. They are part of the local research history.

## Historical Result Notes

Docs under `docs/` record January 2026 experiments. Highlights:

- HPO best trial in `docs/2026-01-26_hpo_and_training_pipeline_updates.md` informed the current tuned `2classSpiders.json`.
- `docs/2026-01-28_sweep_1_0_comparison.md` reports that the hybrid regularized parent pipeline was more stable than the pure playground cost-sensitive regularizer in the documented [1][0] sweep.

Treat these docs as experiment logs. Verify against actual CSV/JSON outputs when making claims about current results.
