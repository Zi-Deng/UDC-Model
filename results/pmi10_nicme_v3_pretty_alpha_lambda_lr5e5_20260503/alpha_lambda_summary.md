# NICME V3 Pretty Alpha/Lambda Grid Summary

Generated: 2026-05-03T17:38:36

- HPO completion: direct resume completed `108/108` grid rows with `0` failed rows.
- The original chain status contains the earlier failed chain entry from run 103; the current post-resume state is the completed `grid_run_ledger.csv` plus this analysis.
- Selection uses validation metrics only.
- Test metrics below are reported after validation-based selection.

## Best Observed Test Row

- run: `95`
- alpha: `0.5`
- cs_lambda: `0.07`
- learning_rate: `5e-05`
- test target-min recall: `0.96875`
- test target-macro recall: `0.984375`
- test normalized ATC: `0.0046875`
- test ATC: `0.046875`
- test balanced accuracy: `0.98125`
- test macro F1: `0.9813859012850947`
- test critical-pair errors: `1`
- validation target-min recall: `1.0`
- validation normalized ATC: `0.0003225806451612903`

Run 95 is the best observed test-ranked row. Run 69 ties run 95 on the main cost-sensitive test metrics and differs only by a very small macro-F1 margin. The validation-selected row remains run 89 below.

## Selected Config

- alpha: `0.4`
- cs_lambda: `0.2`
- learning_rate: `5e-05`
- config: `/mnt/storage/github/NICME/results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/grid-run/configs/0089_3e921dbca43b0608.json`
- metrics: `results/pmi10_sota_convnext_base_test/grid-run_nicme_v3_hybrid_pretty_a0p4_l0p2_lr5e5_s42_05-03_07-41/metrics_20260503_074109_nicme_v3_hybrid.json`

## Validation Metrics

- target-min recall: `1.0`
- target-macro recall: `1.0`
- normalized ATC: `0.0`
- balanced accuracy: `1.0`
- critical-pair errors: `0`

## Test Metrics

- target-min recall: `0.90625`
- target-macro recall: `0.96875`
- normalized ATC: `0.005`
- balanced accuracy: `0.978125`
- critical-pair errors: `1`

## Top 15 By Validation

| Rank | Alpha | Lambda | Val Target-Min | Val Norm. ATC | Val Bal. Acc. | Test Target-Min | Test Norm. ATC | Test Bal. Acc. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4 | 0.2 | 1.0000 | 0.000000 | 1.0000 | 0.9062 | 0.005000 | 0.9781 |
| 2 | 0.5 | 0.07 | 1.0000 | 0.000323 | 0.9968 | 0.9688 | 0.004687 | 0.9812 |
| 3 | 0.4 | 0.09 | 1.0000 | 0.000645 | 0.9935 | 0.9375 | 0.004687 | 0.9812 |
| 4 | 0.04 | 0.02 | 1.0000 | 0.000968 | 0.9903 | 0.9375 | 0.010937 | 0.9625 |
| 5 | 0.09 | 0.09 | 1.0000 | 0.000968 | 0.9903 | 0.9375 | 0.010625 | 0.9656 |
| 6 | 0.4 | 0.03 | 1.0000 | 0.000968 | 0.9903 | 0.9375 | 0.007812 | 0.9719 |
| 7 | 0.6 | 0.04 | 1.0000 | 0.000968 | 0.9903 | 0.9062 | 0.007812 | 0.9781 |
| 8 | 0.3 | 0.2 | 1.0000 | 0.001290 | 0.9871 | 0.9062 | 0.008438 | 0.9719 |
| 9 | 0.2 | 0.09 | 1.0000 | 0.001935 | 0.9806 | 0.9688 | 0.004687 | 0.9812 |
| 10 | 0.3 | 0.09 | 1.0000 | 0.001935 | 0.9806 | 0.9375 | 0.010937 | 0.9625 |
| 11 | 0.5 | 0.2 | 1.0000 | 0.003226 | 0.9677 | 0.9375 | 0.011562 | 0.9563 |
| 12 | 0.06 | 0.2 | 0.9677 | 0.000645 | 0.9935 | 0.9062 | 0.007812 | 0.9781 |
| 13 | 0.09 | 0.05 | 0.9677 | 0.000645 | 0.9935 | 0.9062 | 0.007812 | 0.9781 |
| 14 | 0.1 | 0.1 | 0.9677 | 0.000645 | 0.9935 | 0.9062 | 0.008125 | 0.9750 |
| 15 | 0.1 | 0.3 | 0.9677 | 0.000645 | 0.9935 | 0.8750 | 0.008438 | 0.9719 |
