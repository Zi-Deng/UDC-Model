# PMI-20 NICME Paper SOTA Results

Generated: 2026-05-04T18:01:44

This is the canonical paper-facing PMI-20 result package. It consolidates the completed PMI-20 SOTA baseline suite with the selected NICME alpha `0.5`, lambda `0.1` row from the completed six-candidate NICME rerun.

## Main Result

- NICME is ranked #1 by the predeclared recall-first cost-sensitive composite.
- NICME ties best target-min recall and wins target-macro recall, normalized ATC, and ATC.
- cost-sensitive regularized CE remains slightly higher on balanced accuracy and macro-F1.
- CE + cost-min inference has fewer total critical-pair errors, but with much lower target-min recall.

## Files

- `analysis/aggregate_metrics.csv`
- `analysis/pmi20_sota_table.md`
- `analysis/pmi20_sota_table.tex`
- `analysis/cost_sensitive_winners.md`
- `analysis/claim_audit.md`
- `analysis/method_hyperparameters.md`

## Sources

- Baseline source: `results/pmi20_camera_ready_lr5e5_multiseed_20260504/analysis/aggregate_metrics.csv`
- NICME candidate source: `results/pmi20_nicme_alpha_lambda6_lr5e5_multiseed_20260504/analysis/aggregate_metrics.csv`

The source sweep root has a historical versioned name because it records the original completed run provenance. Current paper-facing text uses `NICME`.
