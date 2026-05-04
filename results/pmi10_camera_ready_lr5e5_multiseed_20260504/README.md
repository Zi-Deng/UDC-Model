# PMI-10 Camera-Ready LR 5e-5 Multi-Seed Results

Generated: 2026-05-03T23:45:54

This folder contains the paper-final fixed-hyperparameter PMI-10 comparison for the NICME NeurIPS table.

## Protocol

- Seeds: `42,43,44`
- Split: `data/prepared/pmi_pills_10_no_cal/splits/balanced`
- Backbone: `timm/convnext_base.fb_in22k_ft_in1k`
- LR profile: `lr5e5`
- Primary NICME: alpha `0.5`, lambda `0.07`, fixed from pilot HPO before this rerun.

## Main Outputs

- `analysis/aggregate_metrics.csv`
- `analysis/neurips_table.md`
- `analysis/neurips_table.tex`
- `analysis/cost_sensitive_winners.md`
- `analysis/paired_deltas_vs_csada.csv`
- `analysis/method_hyperparameters.md`
- `analysis/claim_audit.md`

## Current Composite Leader

- Rank 1: `Menon logit adjustment`

Error bars are sample standard deviations over training seeds on one fixed split.
