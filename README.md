# UDC-Model

Cost-sensitive image classification framework built on PyTorch and HuggingFace Transformers. Trains ResNet-50 and ConvNeXt models with configurable cost matrices that penalize specific misclassification pairs differently.

## Quick Start

```bash
# Set up the environment
micromamba activate ml
pip install -r requirements.txt

# Run training
python scripts/train.py --config config/modelConfig.json
```

## Features

- **Cost-sensitive loss functions**: Cross-entropy, cost-matrix cross-entropy, seesaw loss, and logit-adjusted loss
- **Multiple architectures**: ResNet-50 and ConvNeXt with custom base classes (no HuggingFace model dependency)
- **Flexible data sources**: HuggingFace Hub, Kaggle, local folder, or local CSV datasets
- **Automated sweeps**: Iterate over cost matrix values and collect metrics automatically (bash or Python/Optuna)
- **Hyperparameter optimization**: Optuna-based HPO with grid search
- **Comprehensive evaluation**: Confusion matrices, per-class metrics, JSON/text reports, and visualizations

## Architecture

### Training Pipeline

1. **Config loading**: `parse_HF_args()` reads `--config <path>` and parses JSON into a `ScriptTrainingArguments` dataclass.
2. **Dataset loading**: Dispatches by `dataset_host` (`"huggingface"`, `"kaggle"`, or `"local_folder"` with `local_dataset_format` of `"folder"` or `"csv"`).
3. **Model instantiation**: Creates a ResNet or ConvNeXt model based on `model_type` and loads pretrained weights.
4. **Training**: `CustomTrainer` (extends HuggingFace `Trainer`) injects a custom loss function via `LossFunctions.loss_function(name)`.
5. **Evaluation**: `perform_comprehensive_evaluation()` generates metrics JSON, text reports, and confusion matrix visualizations.

### Loss Function Dispatch

Configure via the `loss_function` field in your JSON config:

| Config Value | Loss Function | Description |
|---|---|---|
| `"cross_entropy"` | Standard CE | Standard cross-entropy loss |
| `"cost_matrix_cross_entropy"` | Cost-matrix CE | Cross-entropy weighted by a per-class cost matrix |
| `"seesaw"` | Seesaw loss | Re-balancing loss for long-tailed distributions |
| `"logit_adjustment"` | Logit-adjusted CE | Label-frequency-aware logit adjustment (V2) |

### Model Implementations

Both architectures use shared base classes defined in `model/__init__.py`:
- `CustomConfig` — configuration dataclass (replaces HuggingFace `PretrainedConfig`)
- `CustomPreTrainedModel` — weight initialization base (replaces HuggingFace `PreTrainedModel`)
- `ImageClassifierOutputWithNoAttention` — output container with `.logits` and `.loss`

## Usage

### Training

```bash
micromamba activate ml
python scripts/train.py --config config/modelConfig.json
```

Results are saved to `results/<output_dir>/<timestamped_run>/` and include:
- `metrics_*.json` — full metrics in JSON format
- `metrics_*.txt` — human-readable report
- `confusion_matrix_*.png` — side-by-side confusion matrices
- `confusion_matrix_detailed_*.png` — detailed matrix with statistics

### Cost Matrix Sweep (Bash)

Iterates over cost values for a specific matrix cell, runs training for each, and generates analysis graphs:

```bash
micromamba activate ml
bash run_cost_matrix_sweep.sh                                    # default config
bash run_cost_matrix_sweep.sh config/sweep_2class_bw_cost.json   # custom config
```

Run multiple sweep configs sequentially:

```bash
bash matrix_sweep_loop.sh
```

See [SWEEP_README.md](SWEEP_README.md) for sweep configuration details.

### Cost Matrix Sweep (Python/Optuna)

Uses Optuna GridSampler with DuckDB storage and Polars for analysis:

```bash
micromamba activate ml
python scripts/cost_matrix_sweep.py --config config/2classSpiders.json
python scripts/cost_matrix_sweep.py --config config/2classSpiders.json --min 0 --max 5 --step 0.5
```

### Hyperparameter Optimization

```bash
micromamba activate ml
python scripts/hpo_search.py --config config/modelConfig.json
```

Uses Optuna to search over learning rate, weight decay, and warmup ratio. Supports both ResNet and ConvNeXt models.

## Config JSON Reference

Key fields in a training config file:

| Field | Type | Description |
|---|---|---|
| `dataset` | string | Dataset name or path |
| `dataset_host` | string | `"huggingface"`, `"kaggle"`, or `"local_folder"` |
| `model` | string | Model name (e.g., `"microsoft/resnet-50"`) |
| `model_type` | string | `"resnet"` or `"convnext"` |
| `weights` | string | Path to pretrained weights file |
| `num_labels` | int | Number of classification classes |
| `learning_rate` | float | Learning rate |
| `num_train_epochs` | int | Number of training epochs |
| `batch_size` | int | Training batch size |
| `loss_function` | string | Loss function name (see dispatch table above) |
| `cost_matrix` | 2D list | Cost matrix (`num_labels x num_labels`); required for cost-matrix losses |
| `local_folder_path` | string | Path to local dataset (when `dataset_host` is `"local_folder"`) |
| `local_dataset_format` | string | `"folder"` or `"csv"` |
| `wandb` | string | `"True"` or `"False"` (string, not boolean) |
| `push_to_hub` | string | `"True"` or `"False"` (string, not boolean) |
| `output_dir` | string | Subdirectory under `results/` for outputs |

## Repository Layout

```
.
├── scripts/
│   ├── __init__.py                   # Package init
│   ├── train.py                      # Main entry point — training + evaluation
│   ├── cost_matrix_sweep.py          # Optuna-based cost matrix sweep
│   └── hpo_search.py                 # Hyperparameter optimization with Optuna
├── model/
│   ├── __init__.py                   # Shared base classes (CustomConfig, CustomPreTrainedModel)
│   ├── ResNet.py                     # ResNet-50 architecture
│   └── convnext.py                   # ConvNeXt architecture
├── utils/
│   ├── __init__.py                   # Package init
│   ├── utils.py                      # Core: arg parsing, dataset loading, metrics, evaluation
│   ├── loss_functions.py             # Loss functions with dispatch
│   ├── image_processor.py            # CustomImageProcessor (replaces HF AutoImageProcessor)
│   ├── extract_metrics.py            # Extracts metrics from results into CSV (bash sweep)
│   └── analyze_cost_matrix_results.py  # Generates graphs from sweep CSV results
├── pyproject.toml                    # Project metadata and ruff configuration
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda/micromamba environment spec
├── config/                           # JSON configs (gitignored)
├── weights/                          # Pretrained weights (gitignored)
└── results/                          # Training outputs (gitignored)
```

## Development

### Linting and Formatting

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
pip install ruff
ruff check .          # lint
ruff check --fix .    # lint with auto-fix
ruff format .         # format
```

Configuration is in `pyproject.toml`.

### Conventions

- Use **Polars** (not pandas) for new dataframe operations.
- Use **DuckDB** for new SQL operations.
- Run all scripts from the repository root.
- Use `python` (not `python3`) inside the micromamba environment.
- `wandb` and `push_to_hub` config fields are strings (`"True"` / `"False"`), not booleans.

## License

See repository for license information.
