# NICME Paper Results Summary

Updated: 2026-05-04

This is the current paper-facing results narrative. Going forward, the method name is **NICME**. The current paper contribution uses the pairwise cost-margin plus expected-cost regularized loss exposed as `nicme_hybrid`; older versioned names are compatibility aliases or historical provenance only.

## Main PMI-20 Result

The canonical paper table is:

- `results/pmi20_nicme_sota_lr5e5_multiseed_20260504/analysis/pmi20_sota_table.md`
- `results/pmi20_nicme_sota_lr5e5_multiseed_20260504/analysis/aggregate_metrics.csv`
- `results/pmi20_nicme_sota_lr5e5_multiseed_20260504/analysis/claim_audit.md`

NICME uses alpha `0.5`, lambda `0.1`, LR `5e-5`, ConvNeXt-base, and seeds `42,43,44` on the balanced full PMI-20 split.

Under the predeclared recall-first cost-sensitive composite, NICME ranks first in the consolidated PMI-20 table:

| Method | Target-min recall | Target-macro recall | Normalized ATC | ATC | Balanced accuracy | Macro-F1 | Critical errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| NICME, alpha 0.5, lambda 0.1 | 0.9167 +/- 0.0180 | 0.9740 +/- 0.0090 | 0.003698 +/- 0.000955 | 0.036979 +/- 0.009547 | 0.9771 +/- 0.0106 | 0.9771 +/- 0.0106 | 1.0000 +/- 1.0000, total 3 |
| AP-CSADA | 0.9167 +/- 0.0180 | 0.9661 +/- 0.0090 | 0.005052 +/- 0.001811 | 0.050521 +/- 0.018110 | 0.9802 +/- 0.0033 | 0.9803 +/- 0.0032 | 2.3333 +/- 1.1547, total 7 |

Supported claims from this table:

- NICME is the best recall-first cost-sensitive tradeoff in the repository PMI-20 comparison.
- NICME ties best target-min recall.
- NICME wins target-macro recall, normalized ATC, and ATC.
- NICME has the lowest critical-pair error count among argmax trained-model rows.

Caveats to keep in the paper:

- AP-CSADA remains slightly higher on balanced accuracy and macro-F1.
- CE + cost-min inference has fewer total critical-pair errors, but with much lower target-min recall.
- The table is a fixed-protocol repository SOTA/baseline comparison, not an external global SOTA claim.
- Error bars are sample standard deviations over three training seeds on one fixed split, not uncertainty over new dataset resamples.

## Supporting PMI-10 Evidence

PMI-10 now serves as supporting sensitivity and robustness evidence rather than the main paper contribution.

- Main multi-seed PMI-10 baseline table: `results/pmi10_camera_ready_lr5e5_multiseed_20260504/analysis/neurips_table.md`
- NICME alpha/lambda top-five rerun: `results/pmi10_nicme_top5_alpha_lambda_lr5e5_multiseed_20260504/analysis/nicme_top5_table.md`
- Historical single-seed PMI-10 HPO: retained as provenance under its generated result root; use it only as supporting evidence.

The PMI-10 roots retain historical generated names where rewriting would damage provenance. Current docs and future runners should use **NICME** without a version suffix.

## Theory And Hyperparameter Memos

- `docs/nicme_vs_csada_theory.pdf`
- `docs/nicme_vs_csada_theory.tex`
- `docs/nicme_hyperparameters.pdf`
- `docs/nicme_hyperparameters.tex`

These memos now use the versionless NICME name and frame the loss as the current method.
