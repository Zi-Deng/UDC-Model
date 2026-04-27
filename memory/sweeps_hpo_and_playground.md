# Sweeps, HPO, And Playground

## Parent Python Cost-Matrix Sweep

`scripts/cost_matrix_sweep.py` sweeps one cost-matrix cell using Optuna `GridSampler`.

Important behavior:

- Input: parent training config JSON, usually `config/2classSpiders.json` or `config/2classSpiders_reg.json`.
- Arguments: `--row`, `--col`, `--min`, `--max`, `--step`, or explicit `--values`.
- For each trial, it deep-copies the base config and sets `config["cost_matrix"][row][col] = cost_value`.
- It runs full parent training by calling `train.main(script_args)`.
- It reads generated metrics from the run directory.
- Objective maximized: class 0 recall.
- Outputs:
  - `sweep_config.json`
  - `sweep_results.csv`
  - `cost_matrix_sweep.duckdb`
  - `graphs/*.html`
  - `graphs/*.png`

The script claims resume support in docs, but the current implementation uses an in-memory `optuna.create_study(...)` with no persistent storage. Existing trials are not automatically resumed from DuckDB.

## Bash Sweep

The bash sweep script exists at:

```text
examples/run_cost_matrix_sweep.sh
```

It:

- reads a sweep config such as `config/sweep_2class_bw_cost.json`
- backs up the base config
- mutates one cost-matrix cell in the base config per iteration
- sets sweep metadata fields in that config
- runs `python scripts/train.py --config <base_config>`
- extracts metrics with `utils/extract_metrics.py`
- optionally updates graphs with `utils/analyze_cost_matrix_results.py`
- restores the original config on exit

The root docs sometimes show `bash run_cost_matrix_sweep.sh`; correct for this checkout is `bash examples/run_cost_matrix_sweep.sh`.

## Hyperparameter Search

`scripts/hpo_search.py` uses HuggingFace `Trainer.hyperparameter_search(backend="optuna")`.

Search space:

- `learning_rate`: log float from `1e-5` to `1e-3`
- `weight_decay`: float from `0.0` to `0.01`
- `per_device_train_batch_size`: categorical `[8, 16, 32]`
- `warmup_ratio`: float from `0.0` to `0.2`
- `lr_scheduler_type`: categorical `["linear", "cosine"]`
- `num_frozen_stages`: integer `0` to `3` for ResNet

It loads dataset and pretrained weights once, creates a fresh model per trial, and uses early stopping with patience 5. Results are saved to:

```text
results/hpo_results/best_hyperparameters.json
```

Historical notes in `docs/2026-01-26_hpo_and_training_pipeline_updates.md` report best trial values near:

- learning rate: `0.00031675827258233213`
- weight decay: `0.009658023670815849`
- batch size: `32`
- warmup ratio: `0.08656072310315271`
- scheduler: `linear`
- frozen stages: `3`

Current `config/2classSpiders.json` rounds these to practical values.

## Sweep Comparison

`scripts/compare_sweeps.py` compares three pipelines:

- parent: `CELogitAdjustmentV2`
- playground: pure cost-sensitive regularized loss (`CE + lambda * CS`)
- hybrid: `CELogitAdjustmentRegularized`

Defaults:

- parent CSV: `results/sweep_cost_0_1/sweep_results.csv`
- playground CSV: `playground/cost_sensitive_loss_classification/results/sweep_cost_ratio/sweep_results.csv`
- hybrid CSV: `results/sweep_cost_0_1_reg/sweep_results.csv`
- output: `results/sweep_comparison`

Outputs:

- `comparison_wide.csv`
- `sweep_comparison.duckdb`
- `comparison_summary.md`
- `graphs/*.html`
- `graphs/*.png` when kaleido works

The generated report includes programmatic conclusions. Treat those conclusions as analysis of specific CSV inputs, not as universal truths.

## Playground Project

Path:

```text
playground/cost_sensitive_loss_classification/
```

This nested project is based on an external cost-sensitive loss classification codebase for diabetic retinopathy. It has its own `.git/`, `CLAUDE.md`, `README.md`, `train.py`, utilities, models, experiments, data CSVs, and results.

Core legacy files:

- `train.py`: diabetic-retinopathy training loop.
- `test_dihedral_tta.py`: test-time augmentation evaluator.
- `test_loop_dihedral.py`: batch evaluator.
- `prepare_training_data.py` and `prepare_test_data.py`: preprocessing scripts requiring additional image-processing dependencies.
- `utils/losses.py`: `CostSensitiveLoss` and `CostSensitiveRegularizedLoss`.
- `utils/get_loaders.py`: CSV image dataset and dataloader helpers.
- `models/get_model.py`: torchvision model factory.

Spider adaptation files:

- `train_spiders.py`: binary Black Widow vs False Widow classifier using the playground loss.
- `cost_ratio_sweep.py`: Optuna sweep over one `M[row][col]` cost value for `train_spiders.py`.

Playground spider class ordering is hardcoded:

```text
0 = Latrodectus_hesperus = Black Widow
1 = Steatoda_grossa = False Widow
```

`train_spiders.py` writes `data/train_spiders.csv`, `data/val_spiders.csv`, and `data/test_spiders.csv` inside the playground directory, then saves experiments under `playground/cost_sensitive_loss_classification/experiments/<save_path>/`.

Default playground spider split fractions:

- validation: 0.15
- test: 0.15

Default playground spider sweep values:

```text
1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100
```

