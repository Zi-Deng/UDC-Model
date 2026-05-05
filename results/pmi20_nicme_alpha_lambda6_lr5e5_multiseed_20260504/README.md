# PMI-20 NICME Six Alpha/Lambda Multi-Seed Results

Generated: 2026-05-04T16:09:48

This folder contains a fixed-protocol PMI-20 rerun for six NICME alpha/lambda candidates.

Status note, 2026-05-04: this is the NICME candidate source suite for the paper table. The consolidated paper-facing comparison uses alpha `0.5`, lambda `0.1` because it provides the best expected-cost performance against the SOTA/baseline rows while tying best target-min recall. See `results/pmi20_nicme_sota_lr5e5_multiseed_20260504/`.

## Protocol

- Seeds: `42,43,44`
- Split: `data/prepared/pmi_pills/splits/balanced`
- Backbone: `timm/convnext_base.fb_in22k_ft_in1k`
- LR profile: `lr5e5`
- Candidates: PMI-10 runs `53, 20, 50, 95, 69` plus added `alpha=0.5, lambda=0.1`.

## Main Outputs

- `analysis/per_seed_metrics.csv`
- `analysis/per_seed_ranks.csv`
- `analysis/aggregate_metrics.csv`
- `analysis/pmi20_nicme_alpha_lambda6_table.md`
- `analysis/pmi20_nicme_alpha_lambda6_table.tex`
- `analysis/cost_sensitive_winners.md`
- `analysis/rank_stability.md`
- `analysis/paired_deltas_vs_run50.csv`
- `analysis/claim_audit.md`
- `analysis/method_hyperparameters.md`

## Current Composite Leader

- Rank 1: `NICME PMI-10 run 95 (alpha=0.5, lambda=0.07)`

Error bars are sample standard deviations over training seeds on one fixed split.
