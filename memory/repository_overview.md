# Repository Overview

## Purpose

NICME (Non-identical Cost Matrix Embeddings) is a Python/PyTorch image classification research repository centered on cost-sensitive learning. The parent project trains custom ResNet and ConvNeXt image classifiers through the HuggingFace `Trainer` API, with configurable loss functions that can penalize particular true-class/predicted-class pairs through a cost matrix.

The current local repository also contains a nested `playground/cost_sensitive_loss_classification/` project. That playground is an adaptation of a MICCAI 2020 cost-sensitive regularization codebase for diabetic retinopathy, plus spider-specific binary-classification scripts and sweep outputs used for comparison against the parent pipeline.

## Stack

- Python: `>=3.12` in `pyproject.toml`; `environment.yml` pins `python=3.12`.
- Core ML: PyTorch, torchvision, HuggingFace `transformers`, `datasets`, `evaluate`, `accelerate`.
- Data and metrics: numpy, pandas, polars, scikit-learn.
- Visualization: matplotlib, seaborn, plotly, kaleido.
- Experiment tooling: Optuna, DuckDB, wandb, kagglehub.
- Formatting/linting: ruff, configured in `pyproject.toml`.

## Main Layout

```text
.
|-- README.md
|-- SWEEP_README.md
|-- pyproject.toml
|-- environment.yml
|-- config/
|-- model/
|-- scripts/
|-- utils/
|-- examples/
|-- data/
|-- weights/
|-- checkpoints/
|-- results/
|-- docs/
|-- playground/cost_sensitive_loss_classification/
`-- memory/
```

## Parent Source Files

- `scripts/train.py`: main parent training and evaluation entry point.
- `scripts/train_reg.py`: wrapper for the regularized hybrid loss configuration.
- `scripts/cost_matrix_sweep.py`: Optuna `GridSampler` sweep over one cost-matrix cell; exports CSV, DuckDB, Plotly outputs.
- `scripts/hpo_search.py`: Optuna hyperparameter search through HuggingFace `Trainer.hyperparameter_search`.
- `scripts/compare_sweeps.py`: compares parent, playground, and hybrid sweep CSVs.
- `nicme/`: package namespace, CLI entry points, and stable import adapters for config, training, and losses.
- `model/ResNet.py`: custom ResNet implementation.
- `model/convnext.py`: custom ConvNeXt implementation.
- `model/__init__.py`: shared HuggingFace-like config/model/output base classes.
- `utils/utils.py`: argument dataclass, dataset loading, preprocessing, metrics, evaluation, result writing.
- `utils/loss_functions.py`: parent loss dispatch and cost-sensitive loss implementations.
- `utils/image_processor.py`: local replacement for HuggingFace `AutoImageProcessor`.
- `utils/extract_metrics.py`: extracts metrics JSON into CSV rows for bash sweeps.
- `utils/analyze_cost_matrix_results.py`: matplotlib/seaborn analysis for bash sweep CSVs.

## Current Git/Local State Notes

- Branch: `main`, tracking `origin/main`.
- The current cleanup branch contains substantial documentation, config, package, and CLI changes for the NICME rename and release-structure pass.
- `.gitignore` ignores heavyweight local artifacts such as `data/`, `checkpoints/`, `results/`, `weights/`, `wandb/`, and `playground/`, plus Python caches, `.ruff_cache/`, and local-only files.
- `docs/`, `examples/`, `config/`, `memory/`, `releases/`, and `nicme/` are intended to be tracked as part of the research-code release.
- Despite `.gitignore`, historical tracked files may still exist under ignored paths such as `weights/` until explicitly removed from version control. Treat `.gitignore` and tracked state separately.
- Root `CLAUDE.md` exists locally but is gitignored. It contains useful historical onboarding notes but is partly stale relative to the current checkout.
