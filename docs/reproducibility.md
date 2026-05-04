# NICME Reproducibility Guide

Updated: 2026-05-04

This guide records the commands and artifact pointers needed to reproduce the current NICME results in this self-contained workspace. Run commands from the repository root unless noted otherwise.

## Environment

```bash
micromamba env update -n ml -f environment.yml
micromamba activate ml
```

Equivalent one-shot form:

```bash
micromamba run -n ml <command>
```

## Current Data And Weights

Required local artifacts include:

- Binary spider data: `data/2_class_black_widows/`
- PMI-10 balanced split: `data/prepared/pmi_pills_10_no_cal/splits/balanced`
- Multiclass balanced splits: `data/prepared/eyepacs_dr/splits/balanced`, `data/prepared/pmi_pills/splits/balanced`
- Pretrained ResNet weights: `weights/pytorch_model.bin`
- Local checkpoints and raw metrics under `checkpoints/` and `results/`

See [artifact_manifest.md](artifact_manifest.md) for sizes, checksums, and external-artifact policy.

## Current PMI-10 HPO And Baselines

The completed current PMI-10 NICME v3 HPO is:

- Root: `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/`
- Ledger: `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/grid_run_ledger.csv`
- Validation ranking: `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/validation_ranked_table.csv`
- Validation-selected config: `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/selected_config.json`
- Post-HPO comparison: `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/post_hpo_sota_comparison.md`

Original chain launcher:

```bash
scripts/launch_pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_chain.sh
```

The completed pre-HPO baseline comparison is:

```bash
scripts/launch_pmi10_sota_pretty_balanced_lr1e5_chain.sh
```

Its current combined summary is `results/pmi10_sota_pretty_balanced_triple_lr_20260503/comparison_summary.md`. The older dual-LR summary is archived because the triple-LR summary plus post-HPO comparison supersede it.

## Current Binary Evidence

The final binary evidence is summarized in [binary_experiment_summary.md](binary_experiment_summary.md). The live result roots are:

- `results/stop3a_balanced_primary/`
- `results/stop3b_imbalance_decoupling/`
- `results/stop4a_backbone_ablation/`
- `results/stop4b_cost_ratio_sensitivity/`

Key machine-readable summaries:

- `results/stop3a_balanced_primary/stop3a_ranked_summary.csv`
- `results/stop3b_imbalance_decoupling/stop3b_imbalance_decoupling_ranked_summary.csv`
- `results/stop4a_backbone_ablation/stop4a_backbone_ablation_ranked_summary.csv`
- `results/stop4b_cost_ratio_sensitivity/spider_convnext/stop4b_spider_cost_ratio_ranked_summary.csv`
- `results/stop4b_cost_ratio_sensitivity/breakhis_dinov3_convnext_lora/stop4b_breakhis_cost_ratio_ranked_summary.csv`

## Current Multiclass State

The current multiclass state is summarized in [multiclass_experiment_summary.md](multiclass_experiment_summary.md). MC2 is complete and MC3 is paused.

Current live artifacts:

- MC2 ledger: `results/multiclass_mc2_official_dinov3_lora_balanced_20260501/mc2/run_ledger.csv`
- MC2 selection: `results/multiclass_mc2_official_dinov3_lora_balanced_20260501/selection/mc2_selection.json`
- MC3 launch commands: `results/multiclass_mc2_official_dinov3_lora_balanced_20260501/selection/mc3_launch_commands.sh`
- MC3 ledger: `results/multiclass_mc3_selected_official_dinov3_lora_balanced_20260501/mc3/run_ledger.csv`

Resume MC3 only after the health checks described in [current_status.md](current_status.md).

## Historical Binary Commands

The original two-class Spider training commands remain valid for historical reproduction, but they are no longer the current best-model workflow:

```bash
python scripts/train.py --config config/nicme_2class_spiders.json
python scripts/train_reg.py --config config/nicme_2class_spiders_regularized.json
python scripts/hpo_search.py --config config/nicme_2class_spiders.json
```

The old best-HPO output moved to `archive/results_docs_cleanup_20260504/results/hpo_results/best_hyperparameters.json`.

Historical cost-matrix sweeps remain under `results/sweep_cost_*` and `results/sweep_comparison*`; they are retained as audit material, not as current PMI-10 or binary-final guidance.

## Validation

```bash
micromamba run -n ml ruff check .
micromamba run -n ml python -m py_compile nicme/*.py scripts/*.py utils/*.py model/*.py
micromamba run -n ml python scripts/validate_release_views.py
```
