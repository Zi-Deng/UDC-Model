# PMI-20 Camera-Ready LR 5e-5 Multi-Seed Results

Generated: 2026-05-04T10:33:40

This folder contains the fixed-hyperparameter full 20-class PMI comparison for the selected SOTA/baseline methods.

Status note, 2026-05-04: this is a source suite for the canonical paper table, not the final paper-facing comparison. Use `results/pmi20_nicme_sota_lr5e5_multiseed_20260504/` for current claims. The NICME row here used the earlier run-50 alpha/lambda setting before the alpha `0.5`, lambda `0.1` paper row was selected.

## Protocol

- Seeds: `42,43,44`
- Split: `data/prepared/pmi_pills/splits/balanced`
- Backbone: `timm/convnext_base.fb_in22k_ft_in1k`
- LR profile: `lr5e5`
- Primary NICME: alpha `0.09`, lambda `0.07`, borrowed from PMI-10 HPO run 50.

## Main Outputs

- `analysis/per_seed_metrics.csv`
- `analysis/aggregate_metrics.csv`
- `analysis/pmi20_sota_table.md`
- `analysis/pmi20_sota_table.tex`
- `analysis/cost_sensitive_winners.md`
- `analysis/paired_deltas_vs_csada.csv`
- `analysis/paired_deltas_vs_menon.csv`
- `analysis/method_hyperparameters.md`
- `analysis/claim_audit.md`

## Current Composite Leader

- Rank 1: `AP-CSADA`

Error bars are sample standard deviations over training seeds on one fixed split.
