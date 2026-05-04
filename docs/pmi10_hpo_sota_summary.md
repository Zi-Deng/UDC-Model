# PMI-10 HPO And SOTA/Baseline Summary

Updated: 2026-05-04

The focused PMI-10 balanced no-calibration track is complete for the LR 5e-5 NICME v3 alpha/lambda HPO.

## Current State

- Completed HPO root: `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/`
- Completion: `108/108` grid rows completed, `0` failed.
- Current HPO guide: `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/README.md`
- Current post-HPO comparison: `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/post_hpo_sota_comparison.md`
- Previous combined baseline summary: `results/pmi10_sota_pretty_balanced_triple_lr_20260503/comparison_summary.md`
- Theory memo: `docs/nicme_v3_vs_csada_theory.pdf`

The archived dual-LR comparison at `archive/results_docs_cleanup_20260504/results/pmi10_sota_pretty_balanced_dual_lr_20260503/` is historical only. Use the triple-LR comparison plus the post-HPO comparison for current interpretation.

## Selection And Best Observed Test Row

| Role | Run | Alpha | Lambda | LR | Target-Min | Norm. ATC | Balanced Acc. | Critical Errors | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Validation-selected | 89 | 0.4 | 0.2 | 5e-05 | 0.90625 | 0.005000 | 0.978125 | 1 | Selected by validation metrics |
| Best observed test row | 95 | 0.5 | 0.07 | 5e-05 | 0.96875 | 0.0046875 | 0.98125 | 1 | Exploratory test-best; validation rank 2 |

Run 95 and run 69 tie on the main cost-sensitive test metrics. Run 95 is listed as the best observed test row because it has a tiny macro-F1 edge.

## Fair Claim

NICME v3 run 95 is the best observed recall-first cost-sensitive tradeoff among the completed repository baselines and completed HPO rows: it ties the previous best target-min recall and target-macro recall, while reducing normalized ATC from `0.015625` to `0.0046875` and improving balanced accuracy from `0.871875` to `0.98125` versus the previous recall-first baseline winner.

The stricter validation-selected result remains run 89. If the project needs a paper-grade final model claim for run 95, confirm it with a prospective selection rule or a fresh multi-seed/nested-validation run.

## Theoretical Interpretation

The NICME v3 vs CSADA memo gives the current theoretical framing: NICME v3 directly shapes pairwise logit margins and clean expected-cost probability mass, while CSADA supplies a boundary-local adversarial signal that may trade away clean balanced accuracy. That memo supports a defensible hypothesis for the observed PMI-10 result; it does not prove universal superiority.
