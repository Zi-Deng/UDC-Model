# Binary Camera-Ready Cost 8/7 LR 5e-5 Multi-Seed Results

Generated: 2026-05-05T04:48:28

This folder contains the fixed-hyperparameter binary SOTA/baseline comparison for Spider and BreaKHis.

## Protocol

- Seeds: `42,43,44`
- Splits: balanced Spider and balanced BreaKHis.
- Backbone: `timm/convnext_base.fb_in22k_ft_in1k`
- LR profile: `lr5e5`
- NICME: alpha `0.5`, lambda `0.1`.
- Spider cost matrix: `[[0,8],[1,0]]`.
- BreaKHis cost matrix: `[[0,1],[7,0]]`.

## Main Outputs

- `analysis/per_seed_metrics.csv`
- `analysis/aggregate_metrics.csv`
- `analysis/binary_sota_table.md`
- `analysis/spider_sota_table.md`
- `analysis/breakhis_sota_table.md`
- `analysis/cost_sensitive_winners.md`
- `analysis/claim_audit.md`

Error bars are sample standard deviations over training seeds on one fixed split per dataset.
- Spider rank 1: `CSADA`
- BreaKHis rank 1: `CSADA`
