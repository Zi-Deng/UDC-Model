# NICME Reproducibility Guide

This guide records the commands needed to reproduce the current NICME results in this self-contained workspace.

## Environment

```bash
micromamba env update -n ml -f environment.yml
micromamba activate ml
```

Run all parent NICME commands from the repository root.

## Data And Weights

Required local artifacts:

- Dataset: `data/2_class_black_widows/`
- Pretrained weights: `weights/pytorch_model.bin`

See `docs/artifact_manifest.md` for sizes, checksums, and intended future external artifact locations.

## Main Training

```bash
python scripts/train.py --config config/nicme_2class_spiders.json
python scripts/train_reg.py --config config/nicme_2class_spiders_regularized.json
```

Equivalent package commands:

```bash
nicme-train --config config/nicme_2class_spiders.json
nicme-train-reg --config config/nicme_2class_spiders_regularized.json
```

## Hyperparameter Optimization

```bash
python scripts/hpo_search.py --config config/nicme_2class_spiders.json
```

Current best HPO output:

- Path: `results/hpo_results/best_hyperparameters.json`
- Best eval accuracy: `0.937037037037037`

## Cost-Matrix Sweeps

Dangerous error direction, Black Widow predicted as False Widow:

```bash
python scripts/cost_matrix_sweep.py \
  --config config/nicme_2class_spiders_regularized.json \
  --row 0 \
  --col 1 \
  --values "1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100" \
  --output-dir results/nicme_sweep_cost_0_1_reg
```

Reverse direction:

```bash
python scripts/cost_matrix_sweep.py \
  --config config/nicme_2class_spiders_regularized.json \
  --row 1 \
  --col 0 \
  --values "1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100" \
  --output-dir results/nicme_sweep_cost_1_0_reg
```

## Comparisons

```bash
python scripts/compare_sweeps.py
```

Comparison inputs:

- Parent LogitAdj: `results/sweep_cost_0_1/sweep_results.csv`
- Hybrid NICME: `results/sweep_cost_0_1_reg/sweep_results.csv`
- Playground baseline: `playground/cost_sensitive_loss_classification/results/sweep_cost_ratio/sweep_results.csv`

## Validation

```bash
micromamba run -n ml ruff check .
micromamba run -n ml python -m py_compile nicme/*.py scripts/*.py utils/*.py model/*.py
micromamba run -n ml python scripts/validate_release_views.py
```

