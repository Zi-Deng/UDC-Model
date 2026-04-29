# NICME

**NICME (Non-identical Cost Matrix Embeddings)** is a PyTorch research codebase for cost-sensitive image classification. The project studies how non-identical, asymmetric cost matrices can steer image classifiers toward safer operating points when different error types have different consequences.

The current experiments focus on binary spider classification:

- Class 0: `Latrodectus_hesperus` / Black Widow
- Class 1: `Steatoda_grossa` / False Widow

The main implementation trains custom ResNet-50-style and ConvNeXt image classifiers with HuggingFace `Trainer`, then evaluates cost-sensitive metrics, confusion matrices, and sweep behavior across cost-matrix values.

The binary-first research extension also supports BreaKHis benign-vs-malignant tumor experiments, calibrated cost-minimum inference, balanced/controlled-imbalance split generation, and Hugging Face / DINOv3 backbones with optional LoRA.

## Quick Start

Run from the repository root:

```bash
micromamba activate ml

# Standard logit-adjustment training
python scripts/train.py --config config/nicme_2class_spiders.json

# Hybrid NICME regularized training
python scripts/train_reg.py --config config/nicme_2class_spiders_regularized.json
```

If installed as a package, the equivalent CLI commands are:

```bash
nicme-train --config config/nicme_2class_spiders.json
nicme-train-reg --config config/nicme_2class_spiders_regularized.json
```

The legacy configs remain supported:

```bash
python scripts/train.py --config config/2classSpiders.json
python scripts/train_reg.py --config config/2classSpiders_reg.json
```

## Environment

The primary environment specification is [environment.yml](environment.yml).

```bash
micromamba env update -n ml -f environment.yml
micromamba activate ml
```

`examples/requirements.txt` is retained as a pip-style dependency reference, but the micromamba environment is the source of truth.

## Main Commands

### Prepare Binary Data

```bash
nicme-prepare-data --dataset spider --input-dir data/2_class_black_widows --output-dir data/prepared/spider
nicme-prepare-data --dataset breakhis --download --extract --raw-dir data/raw/breakhis --output-dir data/prepared/breakhis
```

The BreaKHis archive is large, so the download is explicit and resumable.

### Train

```bash
python scripts/train.py --config config/nicme_2class_spiders.json
python scripts/train_reg.py --config config/nicme_2class_spiders_regularized.json
python scripts/train.py --config config/nicme_spider_balanced_dinov3_vits_lora.json
```

Cost matrices use `C[true_label][predicted_label]` everywhere.

### Cost-Matrix Sweeps

Python/Optuna sweep:

```bash
python scripts/cost_matrix_sweep.py \
  --config config/nicme_2class_spiders_regularized.json \
  --row 0 \
  --col 1 \
  --values "1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100" \
  --output-dir results/nicme_sweep_cost_0_1_reg
```

Legacy bash sweep:

```bash
bash examples/run_cost_matrix_sweep.sh config/nicme_sweep_2class_bw_cost.json
```

### Hyperparameter Optimization

```bash
python scripts/hpo_search.py --config config/nicme_2class_spiders.json
```

### Sweep Comparison

```bash
python scripts/compare_sweeps.py
```

New package entry points mirror these commands:

```bash
nicme-sweep --config config/nicme_2class_spiders_regularized.json --row 0 --col 1 --values "1,2,3"
nicme-hpo --config config/nicme_2class_spiders.json
nicme-compare-sweeps
nicme-run-binary-experiments --base-config config/nicme_spider_balanced_dinov3_vits_lora.json --tier tier1
nicme-experiment-stop --stop 0 --archived-plan docs/experiment_plans/STOP_GATED_MASTER_PLAN.md --split-dir spider_balanced=data/prepared/spider/splits/balanced
```

## Key Results In This Workspace

The current local result summaries are stored under [results/](results/).

| Result | Source | Key finding |
|---|---|---|
| HPO best eval accuracy | `results/hpo_results/best_hyperparameters.json` | `0.9370` eval accuracy with LR `3.17e-4`, batch size `32`, weight decay `0.00966`, warmup `0.0866`, linear scheduler, 3 frozen stages |
| `M[0][1]` hybrid sweep | `results/sweep_cost_0_1_reg/sweep_results.csv` | Best accuracy `0.8967` at cost `5`; best class-0 recall `0.9800` at cost `6` |
| `M[1][0]` hybrid sweep | `results/sweep_cost_1_0_reg/sweep_results.csv` | Best accuracy `0.8967` and best class-0 recall `0.9000` at cost `1` |
| 3-way comparison | `results/sweep_comparison*/comparison_summary.md` | Hybrid LogitAdj+CS avoided the high-cost collapse seen in the playground CE+CS baseline |

See [docs/results_summary.md](docs/results_summary.md) for the paper-facing result table and source paths.

## Repository Layout

```text
config/        Canonical and legacy JSON configs
model/         Custom ResNet and ConvNeXt implementations
nicme/         Package namespace, CLI entry points, and stable import adapters
scripts/       Backward-compatible runnable entry points
utils/         Training, preprocessing, loss, metrics, and analysis utilities
docs/          Reproducibility, artifacts, results, model/data cards, release notes
examples/      Legacy bash sweep and utility examples
results/       Local experiment summaries and generated analysis artifacts
checkpoints/   Local model checkpoints
data/          Local datasets
weights/       Local pretrained weights
playground/    Archived comparison/baseline code
releases/      Curated anonymous and camera-ready release views
memory/        Agent-facing repository memory notes
```

## Methods

NICME currently compares three cost-sensitive training variants:

| Name | Implementation | Description |
|---|---|---|
| Parent LogitAdj | `CELogitAdjustmentV2` | Increases the predicted-class logit for costly misclassified samples before CE-like loss |
| Hybrid NICME regularized | `CELogitAdjustmentRegularized` | Combines LogitAdj with normalized expected-cost regularization and warmup |
| Playground baseline | `CE + cost-sensitive regularization` | Archived comparison implementation from the nested playground project |

Cost matrices use rows as true labels and columns as predicted labels for the LogitAdj and regularized NICME losses. Check the exact loss implementation before comparing against older experimental code.

## Validation

```bash
micromamba run -n ml ruff check .
micromamba run -n ml pytest -q
micromamba run -n ml python -m py_compile nicme/*.py scripts/*.py utils/*.py model/*.py
micromamba run -n ml python scripts/validate_release_views.py
```

## License

No open-source license has been selected yet. See [docs/license_pending.md](docs/license_pending.md).
