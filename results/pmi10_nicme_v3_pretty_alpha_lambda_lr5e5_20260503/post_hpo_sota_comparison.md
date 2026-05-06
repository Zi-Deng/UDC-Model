# Post-HPO SOTA/Baseline Comparison

Generated: 2026-05-03 after the completed LR 5e-5 NICME v3 alpha/lambda HPO.

This comparison consolidates the completed HPO with the previous PMI-10 balanced SOTA/baseline summaries. "SOTA" here means the repository's prior baseline suite, not a new external literature claim.

## Sources

| Source | File |
|---|---|
| Completed HPO ledger | `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/grid_run_ledger.csv` |
| Completed HPO validation ranking | `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/validation_ranked_table.csv` |
| Previous LR 5e-5 baselines | `results/pmi10_sota_pretty_balanced_lr5e5_20260503/analysis/comparison_table.csv` |
| Previous LR 1e-5 baselines | `results/pmi10_sota_pretty_balanced_lr1e5_20260503/analysis/comparison_table.csv` |
| Previous LR 1e-4 baselines | `results/pmi10_sota_pretty_balanced_lr1e4_20260503/analysis/comparison_table.csv` |
| Previous combined summary | `results/pmi10_sota_pretty_balanced_triple_lr_20260503/comparison_summary.md` |

## Metric Convention

- Target-min recall is the minimum recall over the cared PMI classes.
- Normalized ATC is average test cost divided by the maximum off-diagonal cost; lower is better.
- Critical errors count mistakes on the configured high-cost PMI confusion pairs.
- The prior comparison ranking is recall-first: target-min recall descending, then normalized ATC ascending, then balanced accuracy descending, then critical-pair errors ascending.

## Current Top Rows Under Recall-First Cost-Sensitive Ranking

| Rank | Source | Method / Run | LR | Alpha | Lambda | Decision | Target-Min | Target-Macro | Norm. ATC | ATC | Bal. Acc. | Macro F1 | Critical Errors |
|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | HPO LR 5e-5 | NICME v3 run 95 | 5e-05 | 0.5 | 0.07 | argmax | 0.9688 | 0.9844 | 0.004688 | 0.046875 | 0.9812 | 0.981386 | 1 |
| 2 | HPO LR 5e-5 | NICME v3 run 69 | 5e-05 | 0.2 | 0.09 | argmax | 0.9688 | 0.9844 | 0.004688 | 0.046875 | 0.9812 | 0.981293 | 1 |
| 3 | Previous baseline | CSADA, LR 5e-5 profile | 1e-05 | n/a | n/a | argmax | 0.9688 | 0.9844 | 0.015625 | 0.156250 | 0.8719 | 0.851092 | 1 |
| 4 | HPO LR 5e-5 | NICME v3 run 20 | 5e-05 | 0.06 | 0.03 | argmax | 0.9375 | 0.9766 | 0.004688 | 0.046875 | 0.9812 | 0.981339 | 1 |
| 5 | HPO LR 5e-5 | NICME v3 run 50 | 5e-05 | 0.09 | 0.07 | argmax | 0.9375 | 0.9766 | 0.004688 | 0.046875 | 0.9812 | 0.981339 | 1 |
| 6 | Previous baseline | cost-sensitive regularized CE, LR 1e-4 profile | 1e-05 | n/a | n/a | argmax | 0.9375 | 0.9766 | 0.005000 | 0.050000 | 0.9781 | 0.978349 | 1 |

## Previous Best Baseline Versus HPO Run 95

| Metric | Previous best recall-first baseline: CSADA | HPO run 95: NICME v3 alpha 0.5 lambda 0.07 | Direction |
|---|---:|---:|---|
| Target-min recall | 0.96875 | 0.96875 | tie |
| Target-macro recall | 0.984375 | 0.984375 | tie |
| Normalized ATC | 0.015625 | 0.0046875 | HPO lower by 0.0109375, a 70.0% reduction |
| ATC | 0.15625 | 0.046875 | HPO lower by 0.109375, a 70.0% reduction |
| Balanced accuracy | 0.871875 | 0.98125 | HPO higher by 0.109375 |
| Macro F1 | 0.851092 | 0.981386 | HPO higher by 0.130294 |
| Critical-pair errors | 1 | 1 | tie |

## Important Caveats

- HPO run 95 is the best observed test row, but the validation-selected config is run 89.
- Run 95 and run 69 are tied on all main cost-sensitive test metrics: target-min recall, target-macro recall, normalized ATC, ATC, balanced accuracy, accuracy, and critical-pair errors. Run 95 only wins by a tiny macro-F1 edge.
- If the sole objective is minimum test normalized ATC or zero critical-pair errors, `ce_cost_min_inference_pretty` at LR 1e-4 remains better with normalized ATC `0.0025` and `0` critical errors, but it does so with lower target-min recall `0.90625`.
- The strongest fair claim is that NICME v3 run 95 is the best observed recall-first cost-sensitive tradeoff among the completed repository baselines and the completed HPO, not that it universally minimizes every cost metric.
