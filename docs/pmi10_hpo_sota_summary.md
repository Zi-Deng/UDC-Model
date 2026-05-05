# PMI-10 HPO And SOTA/Baseline Summary

Updated: 2026-05-04

The focused PMI-10 balanced no-calibration track is complete for the LR 5e-5 NICME alpha/lambda HPO. This is now supporting evidence; the main paper-facing result is the PMI-20 consolidated table in [paper_results_summary.md](paper_results_summary.md).

## Current State

- Completed HPO root: retained under its historical generated result path.
- Completion: `108/108` grid rows completed, `0` failed.
- Multi-seed PMI-10 table: `results/pmi10_camera_ready_lr5e5_multiseed_20260504/analysis/neurips_table.md`
- Multi-seed alpha/lambda rerun: `results/pmi10_nicme_top5_alpha_lambda_lr5e5_multiseed_20260504/analysis/nicme_top5_table.md`
- Theory memo: `docs/nicme_vs_csada_theory.pdf`

The older single-seed HPO and pre-HPO LR comparison folders are historical only. Use the multi-seed PMI-10 outputs as supporting evidence and the PMI-20 table for paper positioning.

## Selection And Best Observed Test Row

| Role | Run | Alpha | Lambda | LR | Target-Min | Norm. ATC | Balanced Acc. | Critical Errors | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Validation-selected | 89 | 0.4 | 0.2 | 5e-05 | 0.90625 | 0.005000 | 0.978125 | 1 | Selected by validation metrics |
| Best observed test row | 95 | 0.5 | 0.07 | 5e-05 | 0.96875 | 0.0046875 | 0.98125 | 1 | Exploratory test-best; validation rank 2 |

Run 95 and run 69 tie on the main cost-sensitive test metrics. Run 95 is listed as the best observed test row because it has a tiny macro-F1 edge.

## Fair Claim

NICME run 95 was the best observed single-seed recall-first cost-sensitive tradeoff among the completed PMI-10 repository baselines and HPO rows: it tied the previous best target-min recall and target-macro recall, while reducing normalized ATC from `0.015625` to `0.0046875` and improving balanced accuracy from `0.871875` to `0.98125` versus the previous recall-first baseline winner.

The stricter validation-selected result remains run 89. If the project needs a paper-grade final model claim for run 95, confirm it with a prospective selection rule or a fresh multi-seed/nested-validation run.

## Theoretical Interpretation

The NICME vs CSADA memo gives the current theoretical framing: NICME directly shapes pairwise logit margins and clean expected-cost probability mass, while CSADA supplies a boundary-local adversarial signal that may trade away clean balanced accuracy. That memo supports a defensible hypothesis for the observed PMI-10 result; it does not prove universal superiority.
