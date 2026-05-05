# PMI-10 NICME Top-5 Alpha/Lambda Multi-Seed Results

Generated: 2026-05-04T04:23:32

This folder contains a fixed-protocol robustness rerun for the top five observed NICME alpha/lambda settings from the LR 5e-5 pilot HPO.

## Protocol

- Seeds: `42,43,44`
- Split: `data/prepared/pmi_pills_10_no_cal/splits/balanced`
- Backbone: `timm/convnext_base.fb_in22k_ft_in1k`
- LR profile: `lr5e5`
- Candidate HPO runs: `95, 69, 20, 50, 53`

## Main Outputs

- `analysis/per_seed_metrics.csv`
- `analysis/aggregate_metrics.csv`
- `analysis/nicme_top5_table.md`
- `analysis/nicme_top5_table.tex`
- `analysis/rank_stability.md`
- `analysis/paired_deltas_vs_run95.csv`
- `analysis/claim_audit.md`
- `analysis/method_hyperparameters.md`

## Current Composite Leader

- Rank 1: `NICME run 53 (alpha=0.09, lambda=0.2)`

Error bars are sample standard deviations over training seeds on one fixed split.
